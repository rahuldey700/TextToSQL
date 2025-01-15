import os
import streamlit as st
import duckdb
from dotenv import load_dotenv

# LangChain references
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BaseChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Helpers
from helpers import (
    create_table_from_csv,
    create_table_from_excel,
    get_schema_description,
    execute_sql_and_return_results,
    apply_fuzzy_matching_to_query,
    data_cleaning_tool,
    semantic_column_selection
)

########################################
#   CUSTOM CALLBACK HANDLER (DISPLAY)  #
########################################

class StreamlitCallbackHandler(BaseCallbackHandler):
    """
    Streams the agent's step-by-step output into the Streamlit UI.
    Displays chain-of-thought, final queries, and final answers
    in a chat-like "Agent Interaction" section.
    """
    def __init__(self, container):
        super().__init__()
        self.container = container  # The container is the "Agent Interaction" chat box

    def on_tool_start(self, serialized, input_str: str, **kwargs):
        tool_name = serialized.get('name','unknown_tool')
        self.container.markdown(f"**Tool Called**: `{tool_name}`")
        if tool_name == "run_sql_query":
            self.container.markdown("**Proposed SQL**:")
            sql_clean = input_str.replace('```sql','').replace('```','').strip()
            self.container.code(sql_clean, language="sql")
        else:
            self.container.markdown(f"**Tool Input**: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        if "### SQL Query" in output:
            parts = output.split("### Results")
            if len(parts)>1:
                sql_part = parts[0].replace("### SQL Query","").strip()
                results_part = parts[1].strip()

                self.container.markdown("**Final SQL Query**:")
                self.container.code(sql_part, language="sql")
                self.container.markdown("**Query Results**:")
                self.container.markdown(results_part)
            else:
                self.container.markdown("**Tool Output**:")
                self.container.markdown(output)
        else:
            self.container.markdown("**Tool Output**:")
            self.container.markdown(output)
        self.container.markdown("---")

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.container.markdown("**Agent**: Starting chain-of-thought...")

    def on_chain_end(self, outputs, **kwargs):
        final = outputs.get('output','')
        self.container.markdown(f"**Agent Final Answer**: {final}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.container.markdown("*LLM thinking...*")

    def on_llm_end(self, response, **kwargs):
        text = response.generations[0][0].text
        if text.strip():
            self.container.markdown(f"*LLM Partial:* {text}")

    def on_agent_action(self, action, **kwargs):
        self.container.markdown(f"**Agent Action**:\n{action.log}")

    def on_text(self, text: str, **kwargs):
        self.container.markdown(f"**Agent Note**: {text}")


########################################
#   TIMESTAMP CAST + TABLE FALLBACK    #
########################################

def attempt_timestamp_cast(original_query: str) -> str:
    """
    If user tries `AVG(timestamp_column)`, replace with `AVG(CAST(timestamp_column AS DOUBLE))`.
    """
    import re
    if "avg(" not in original_query.lower():
        return original_query
    pattern = r"AVG\((.*?)\)"
    match = re.search(pattern, original_query, re.IGNORECASE)
    if match:
        col = match.group(1).strip()
        cast_expr = f"AVG(CAST({col} AS DOUBLE))"
        cast_query = re.sub(pattern, cast_expr, original_query, flags=re.IGNORECASE)
        return cast_query
    return original_query

def attempt_table_fallback(original_query: str) -> str:
    """
    If user references 'FROM table' or 'FROM your_table' but there's only 1 real table,
    we replace it with that table name. Case-insensitive matching.
    """
    q_lower = original_query.lower()

    # Filter out 'demo_data'
    user_tables = [t for t in st.session_state["loaded_tables"] if t!="demo_data"]
    if len(user_tables)==1:
        # Find the position of 'FROM' and the word after it
        from_pos = q_lower.find('from ')
        if from_pos != -1:
            # Split the query into before FROM and after FROM
            before_from = original_query[:from_pos]
            after_from = original_query[from_pos+5:]  # +5 to skip 'from '
            
            # Replace the table name if it matches our patterns
            words_after_from = after_from.lower().split()
            if words_after_from and words_after_from[0] in ['table', 'your_table']:
                # Preserve the rest of the query after the table name
                rest_of_query = ' '.join(words_after_from[1:])
                return f"{before_from}FROM {user_tables[0]} {rest_of_query}".strip()
    
    return original_query


########################################
#        MAIN SQL RUN TOOL           #
########################################

def run_sql_query_tool(sql_query: str) -> str:
    """
    Execute the SQL query. 
    1) If aggregator is TIMESTAMP, cast it to DOUBLE 
    2) If user references "FROM table"/"FROM your_table" but there's 1 real table, fix it
    3) If error, do fuzzy matching once
    """
    conn = st.session_state["duckdb_conn"]

    # 1) If user references 'table', fallback
    fixed_query = attempt_table_fallback(sql_query)

    # 2) Execute
    df, err = execute_sql_and_return_results(conn, fixed_query)
    if err:
        # Specifically check for TIMESTAMP aggregator
        if "No function matches the given name and argument types 'avg(TIMESTAMP_NS)'" in err:
            # Attempt cast
            cast_query = attempt_timestamp_cast(fixed_query)
            if cast_query != fixed_query:
                df2, err2 = execute_sql_and_return_results(conn, cast_query)
                if not err2:
                    prev = df2.to_markdown(index=False) if not df2.empty else "No results found."
                    return (
                        "### SQL Query\n" + cast_query + "\n"
                        + "### Results\n" + prev
                    )
                else:
                    return f"**SQL Error** after cast: {err2}"

        # 3) Fuzzy matching
        revised = apply_fuzzy_matching_to_query(fixed_query, st.session_state["loaded_tables"], conn)
        if revised != fixed_query:
            df2, err2 = execute_sql_and_return_results(conn, revised)
            if not err2:
                preview2 = df2.to_markdown(index=False) if not df2.empty else "No results found."
                return (
                    "### SQL Query\n"
                    + revised + "\n"
                    + "### Results\n"
                    + preview2
                )
            else:
                return f"**SQL Error** after fuzzy-correction: {err2}"

        return f"**SQL Error**: {err}"
    else:
        # success
        preview = df.to_markdown(index=False) if not df.empty else "No results found."
        return (
            "### SQL Query\n"
            + fixed_query + "\n"
            + "### Results\n"
            + preview
        )


########################################
#  ALL OTHER TOOLS & PROMPT/PARSER    #
########################################

def get_schema_tool(_: str) -> str:
    if not st.session_state.loaded_tables:
        return "No tables loaded."
    return get_schema_description(st.session_state["duckdb_conn"], st.session_state["loaded_tables"])

def rename_column_tool(args: str) -> str:
    conn = st.session_state["duckdb_conn"]
    try:
        data = {}
        for p in args.split(","):
            k, v = p.split("=")
            data[k.strip()] = v.strip()
        table = data.get("table")
        old_col = data.get("old_col")
        new_col = data.get("new_col")

        sql = f'ALTER TABLE "{table}" RENAME COLUMN "{old_col}" TO "{new_col}"'
        conn.execute(sql)
        return f"Renamed column '{old_col}' -> '{new_col}' in '{table}'."
    except Exception as e:
        return f"Failed to rename column: {e}"

def alter_column_type_tool(args: str) -> str:
    conn = st.session_state["duckdb_conn"]
    try:
        data = {}
        for p in args.split(","):
            k, v = p.split("=")
            data[k.strip()] = v.strip()

        table = data.get("table")
        column = data.get("column")
        new_type = data.get("new_type")
        
        sql = f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET DATA TYPE {new_type}'
        conn.execute(sql)
        return f"Changed type of '{column}' in '{table}' to {new_type}."
    except Exception as e:
        return f"Failed to alter column type: {e}"

def list_tables_tool(_: str) -> str:
    if "loaded_tables" not in st.session_state or not st.session_state["loaded_tables"]:
        return "No tables loaded."
    return ", ".join(st.session_state["loaded_tables"])

def data_cleaning_tool_wrapper(args: str) -> str:
    try:
        kv_pairs = [x.strip() for x in args.split(",")]
        data = {}
        for pair in kv_pairs:
            if "=" not in pair:
                return f"Invalid argument: {pair}"
            k, v = pair.split("=",1)
            data[k.strip()] = v.strip()

        table = data.get("table")
        column = data.get("column")
        strat = data.get("strategy","numeric_only")

        if not table or not column:
            return "Usage: table=<tbl>, column=<col>, strategy=<option>"
        return data_cleaning_tool(st.session_state["duckdb_conn"], table, column, strat)
    except Exception as e:
        return f"data_cleaning_tool error: {e}"

def semantic_col_selector_tool(args: str) -> str:
    try:
        user_q = ""
        if "=" in args:
            data = {}
            for p in args.split(","):
                if "=" not in p:
                    return f"Invalid param: {p}"
                k, v = p.split("=",1)
                data[k.strip()] = v.strip()
            user_q = data.get("user_question","")
        else:
            user_q = args.strip()

        if not user_q:
            return "No question provided."

        relevant = semantic_column_selection(
            st.session_state["duckdb_conn"],
            user_q,
            st.session_state["loaded_tables"]
        )
        if relevant:
            return f"Found relevant columns: {', '.join(relevant)}"
        else:
            return "No direct column match found. Possibly clarify with user or fallback."
    except Exception as e:
        return f"Error in semantic_col_selector_tool: {e}"

class SQLAgentPrompt(BaseChatPromptTemplate):
    def format_messages(self, **kwargs):
        user_tables = [t for t in st.session_state["loaded_tables"] if t != "demo_data"]
        system_text = f"""
You are an AI that can answer user questions about data in DuckDB. 
Tools:
 - run_sql_query
 - get_schema
 - rename_column
 - alter_column_type
 - list_tables
 - semantic_col_selector
 - data_cleaning_tool

If there's a TIMESTAMP aggregator mismatch, cast it automatically. 
If question is unclear, ask user. 
If multi-file, reason about the correct table. 
Always show final SQL & results if run_sql_query.
"""
        user_text = kwargs["input"]
        return [
            {"role":"system","content":system_text},
            {"role":"user","content":user_text}
        ]

class SQLAgentOutputParser(BaseOutputParser):
    def parse(self, text: str):
        ltext = text.lower()
        if "action:" in ltext and "action input:" in ltext:
            lines = text.splitlines()
            action_line = None
            input_line = None
            for l in lines:
                if "action:" in l.lower():
                    action_line = l
                if "action input:" in l.lower():
                    input_line = l
            if action_line and input_line:
                tool_name = action_line.split(":",1)[1].strip()
                tool_input = input_line.split(":",1)[1].strip()
                return AgentAction(tool=tool_name, tool_input=tool_input, log=text)
        return AgentFinish({"output": text}, text)

def build_agent(callbacks=None):
    llm = ChatOpenAI(
        api_key=st.session_state["api_key"],
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    tools = [
        Tool(name="run_sql_query", func=run_sql_query_tool, description="Execute queries on DuckDB"),
        Tool(name="get_schema", func=get_schema_tool, description="Show DB schema"),
        Tool(name="rename_column", func=rename_column_tool, description="Rename a column"),
        Tool(name="alter_column_type", func=alter_column_type_tool, description="Change a column's data type"),
        Tool(name="list_tables", func=list_tables_tool, description="List loaded tables"),
        Tool(name="semantic_col_selector", func=semantic_col_selector_tool, description="Find relevant columns"),
        Tool(name="data_cleaning_tool", func=data_cleaning_tool_wrapper, description="Clean columns with numeric_only, etc."),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You can fix columns if TIMESTAMP aggregator fails by casting to DOUBLE. 
Always finalize with run_sql_query if user wants data.
"""),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user","{input}")
    ])
    
    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callbacks=callbacks,
        verbose=True
    )

####################################
#        STREAMLIT FRONT END       #
####################################

def main():
    load_dotenv()

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.getenv("OPENAI_API_KEY","")
    if not st.session_state["api_key"]:
        st.error("No OPENAI_API_KEY found.")
        return

    st.set_page_config(page_title="Agent with Chat UI", layout="wide")

    if "duckdb_conn" not in st.session_state:
        st.session_state["duckdb_conn"] = duckdb.connect(database=":memory:")
    if "loaded_tables" not in st.session_state:
        st.session_state["loaded_tables"] = []

    st.title("Agent Interaction")

    # Data Load Section
    st.subheader("Data Loading")
    if st.button("Load sample_data.csv as demo_data"):
        try:
            create_table_from_csv(st.session_state["duckdb_conn"], "sample_data.csv", "demo_data")
            if "demo_data" not in st.session_state["loaded_tables"]:
                st.session_state["loaded_tables"].append("demo_data")
            st.success("demo_data loaded!")
        except FileNotFoundError:
            st.error("sample_data.csv not found.")

    files = st.file_uploader("Upload CSV/XLSX", accept_multiple_files=True, type=["csv","xlsx"])
    if files:
        for f in files:
            name = f.name.replace(" ","_").replace(".","_").lower()
            if f.name.endswith(".csv"):
                create_table_from_csv(st.session_state["duckdb_conn"], f, name)
            else:
                create_table_from_excel(st.session_state["duckdb_conn"], f, name)
            if name not in st.session_state["loaded_tables"]:
                st.session_state["loaded_tables"].append(name)
        st.success("Files loaded successfully.")

    if st.session_state["loaded_tables"]:
        st.markdown(f"**Loaded Tables**: {st.session_state['loaded_tables']}")
        pick = st.selectbox("Preview a table:", st.session_state["loaded_tables"])
        if pick:
            with st.expander(f"Preview {pick}"):
                sm = st.session_state["duckdb_conn"].execute(f"SELECT * FROM {pick} LIMIT 5").fetchdf()
                st.dataframe(sm)

    # Agent Interaction (Chat UI)
    st.subheader("Agent Interaction")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    conversation_container = st.container()
    for msg in st.session_state["chat_history"]:
        if msg["role"]=="user":
            conversation_container.markdown(f"**You**: {msg['content']}")
        else:
            conversation_container.markdown(f"**Agent**: {msg['content']}")

    debug_expander = st.expander("Agent Thought Process", expanded=False)
    debug_container = debug_expander.container()

    def submit_chat():
        user_msg = st.session_state.get("user_input","").strip()
        if not user_msg:
            st.warning("Please type something.")
            return
        st.session_state["chat_history"].append({"role":"user","content":user_msg})
        st.session_state["user_input"] = ""

        conversation_container.markdown(f"**You**: {user_msg}")

        cb_handler = StreamlitCallbackHandler(debug_container)
        agent = build_agent(callbacks=[cb_handler])

        with st.spinner("Thinking..."):
            try:
                ans_dict = agent.invoke({"input":user_msg})
                ans = ans_dict.get("output", "") if isinstance(ans_dict, dict) else ans_dict
                if ans.strip():
                    st.session_state["chat_history"].append({"role":"assistant","content":ans})
                    conversation_container.markdown(f"**Agent**: {ans}")
            except Exception as e:
                st.error(f"Agent error: {e}")

    st.text_input("Your message", key="user_input", on_change=submit_chat)

if __name__ == "__main__":
    main()