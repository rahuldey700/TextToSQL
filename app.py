import os
import streamlit as st
import duckdb
from dotenv import load_dotenv
from typing import List
import re

#streamlit?

# LangChain references
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BaseChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Helpers
from helpers import (
    create_table_from_csv,
    create_table_from_excel,
    get_schema_description,
    execute_sql_and_return_results,
    apply_fuzzy_matching_to_query,
    data_cleaning_tool,
    semantic_column_selection,
    get_all_columns
)

####################################
#    CUSTOM CALLBACK HANDLER       #
####################################

class StreamlitCallbackHandler(BaseCallbackHandler):
    """
    Streams the agent's step-by-step output into the Streamlit UI.
    Displays chain-of-thought, final queries, and final answers
    in a chat-like "Agent Interaction" section.
    """
    def __init__(self, container, conversation_container):
        super().__init__()
        self.container = container  # The container is the "Agent Interaction" chat box
        self.conversation_container = conversation_container  # Main conversation area
        self.last_sql_query = None  # Store the last SQL query

    def on_tool_start(self, serialized, input_str: str, **kwargs):
        tool_name = serialized.get('name','unknown_tool')
        self.container.markdown(f"**Tool Called**: `{tool_name}`")
        if tool_name == "run_sql_query":
            # Display the proposed SQL
            self.container.markdown("**Proposed SQL**:")
            sql_clean = input_str.replace('```sql','').replace('```','').strip()
            self.container.code(sql_clean, language="sql")
        else:
            self.container.markdown(f"**Tool Input**: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        self.last_sql_query = None  # Reset SQL query
        if "### SQL Query" in output:
            # Typical format: "### SQL Query\n ... ### Results\n ..."
            parts = output.split("### Results")
            if len(parts)>1:
                sql_part = parts[0].replace("### SQL Query","").strip()
                results_part = parts[1].strip()
                
                # Store the SQL query for final display
                self.last_sql_query = sql_part

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
        # If we have a SQL query, include it in the final response
        if self.last_sql_query:
            final = f"**Final SQL Query**:\n```sql\n{self.last_sql_query}\n```\n\n**Results**:\n{final}"
        self.conversation_container.markdown(final)

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


####################################
#        TOOL DEFINITIONS          #
####################################

def parse_table_names(sql_query: str) -> List[str]:
    """
    Parse the table names from a simple SQL query by looking for 'FROM <table>'.
    This is a naive approach but works for many straightforward queries.
    """
    # Use a case-insensitive regex to capture anything after FROM up until whitespace or punctuation
    pattern = r'(?i)\bFROM\s+["`]?([\w]+)["`]?'
    tables = re.findall(pattern, sql_query)
    # Return unique tables
    return list(set(t.strip() for t in tables))


def run_sql_query_tool(sql_query: str) -> str:
    """
    Execute the SQL query on DuckDB.
    If aggregator mismatch on a timestamp, attempt casting.
    If error persists, try fuzzy matching columns.
    """
    conn = st.session_state["duckdb_conn"]
    loaded_tables = st.session_state.get("loaded_tables", [])

    # Identify which table(s) the query references
    referenced_tables = parse_table_names(sql_query)
    if not referenced_tables and len(loaded_tables) == 1:
        # If user didn't specify any table name, but exactly one table is loaded, assume that table
        referenced_tables = [loaded_tables[0]]

    # Check if all referenced tables are actually loaded
    for tbl in referenced_tables:
        if tbl not in loaded_tables:
            return (
                f"**Table '{tbl}' is not loaded.**\n"
                "If you need that data, please load it, or correct your table name."
            )

    # Removed the manual unknown-token logic here
    # Let DuckDB itself validate columns or throw errors.

    # Attempt the query
    df, err = execute_sql_and_return_results(conn, sql_query)
    if err:
        # If aggregator mismatch on timestamp, attempt automatic CAST
        if "No function matches the given name and argument types 'avg(TIMESTAMP_NS)'" in err:
            cast_query = attempt_timestamp_cast(sql_query)
            if cast_query != sql_query:
                df2, err2 = execute_sql_and_return_results(conn, cast_query)
                if not err2:
                    prev = df2.to_markdown(index=False) if not df2.empty else "No results found."
                    return "### SQL Query\n" + cast_query + "\n### Results\n" + prev
                else:
                    return f"**SQL Error** after casting timestamp: {err2}"

        # Attempt fuzzy matching fallback if DuckDB complained about unknown column
        # or other syntax issues
        revised = apply_fuzzy_matching_to_query(sql_query, loaded_tables, conn)
        if revised != sql_query:
            df2, err2 = execute_sql_and_return_results(conn, revised)
            if not err2:
                preview2 = df2.to_markdown(index=False) if not df2.empty else "No results found."
                return "### SQL Query\n" + revised + "\n### Results\n" + preview2
            else:
                return f"**SQL Error** after fuzzy-correction: {err2}"

        return f"**SQL Error**: {err}"

    # If no error, return results
    preview = df.to_markdown(index=False) if not df.empty else "No results found."
    return "### SQL Query\n" + sql_query + "\n### Results\n" + preview

def attempt_timestamp_cast(original_query: str) -> str:
    """
    Simple approach: find something like `AVG(column)`
    and replace with `AVG(CAST(column AS DOUBLE))`.
    Only do it for columns that might be TIMESTAMP.
    """
    # naive approach: replace "avg(" => "avg(cast(" and add as double
    # but let's only do it if we see "avg(" in the query
    if "avg(" not in original_query.lower():
        return original_query

    # Identify the substring between avg( and next ) (very naive)
    import re
    pattern = r"AVG\((.*?)\)"
    match = re.search(pattern, original_query, re.IGNORECASE)
    if match:
        col = match.group(1).strip()
        # produce "AVG(CAST(col AS DOUBLE))"
        cast_expr = f"AVG(CAST({col} AS DOUBLE))"
        cast_query = re.sub(pattern, cast_expr, original_query, flags=re.IGNORECASE)
        return cast_query
    return original_query

def get_schema_tool(_: str) -> str:
    if not st.session_state.loaded_tables:
        return "No tables loaded."
    return get_schema_description(st.session_state["duckdb_conn"], st.session_state["loaded_tables"])

def rename_column_tool(args: str) -> str:
    # same as your code
    conn = st.session_state["duckdb_conn"]
    try:
        data = {}
        for p in args.split(","):
            k, v = p.split("=")
            data[k.strip()] = v.strip()
        table = data.get("table")
        old_col = data.get("old_col")
        new_col = data.get("new_col")

        rename_sql = f'ALTER TABLE "{table}" RENAME COLUMN "{old_col}" TO "{new_col}"'
        conn.execute(rename_sql)
        return f"Renamed column '{old_col}' -> '{new_col}' in table '{table}'."
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
        return f"Changed type of '{column}' in table '{table}' to {new_type}."
    except Exception as e:
        return f"Failed to alter column type: {e}"

def list_tables_tool(_: str) -> str:
    if "loaded_tables" not in st.session_state or not st.session_state["loaded_tables"]:
        return "No tables loaded."
    return ", ".join(st.session_state["loaded_tables"])

def data_cleaning_tool_wrapper(args: str) -> str:
    # same as your code
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
    # same as your code
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

####################################
#     AGENT PROMPT & OUTPUT PARSER 
####################################

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
        model_name="gpt-4o",
        temperature=0,
        callbacks=callbacks
    )

    # Initialize memory if it doesn't exist in session state
    if "agent_memory" not in st.session_state:
        st.session_state["agent_memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    tools = [
        Tool(name="run_sql_query", func=run_sql_query_tool, description="Execute queries on DuckDB"),
        Tool(name="get_schema", func=get_schema_tool, description="Show DB schema"),
        Tool(name="rename_column", func=rename_column_tool, description="Rename a column"),
        Tool(name="alter_column_type", func=alter_column_type_tool, description="Change column type"),
        Tool(name="list_tables", func=list_tables_tool, description="List loaded tables"),
        Tool(name="semantic_col_selector", func=semantic_col_selector_tool, description="Find relevant columns"),
        Tool(name="data_cleaning_tool", func=data_cleaning_tool_wrapper, description="Clean columns with numeric_only, etc."),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You can fix columns if TIMESTAMP aggregator fails by casting to DOUBLE. 
Currently loaded tables: {', '.join(st.session_state["loaded_tables"])}
Always finalize with run_sql_query if user wants data.
Remember previous conversations and use that context when appropriate.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
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
        memory=st.session_state["agent_memory"],
        callbacks=callbacks,
        verbose=True
    )

####################################
#        STREAMLIT FRONT END       #
####################################

def main():
    load_dotenv()

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.getenv("OPENAI_API_KEY")
    if not st.session_state["api_key"]:
        st.error("No OPENAI_API_KEY found in environment variables.")
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
            # Strip file extension and any non-alphanumeric characters
            base_name = os.path.splitext(f.name)[0]  # Remove extension first
            name = ''.join(c.lower() for c in base_name if c.isalnum())
            
            # Ensure name starts with a letter (SQL requirement)
            if name and not name[0].isalpha():
                name = 'tbl_' + name
            
            # Fallback if name is empty after sanitization
            if not name:
                name = 'table_' + str(len(st.session_state["loaded_tables"]))
            
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
                sm = st.session_state["duckdb_conn"].execute(f'SELECT * FROM "{pick}" LIMIT 5').fetchdf()
                st.dataframe(sm)

    # Agent Interaction (Chat UI)
    st.subheader("Agent Interaction")

    # We'll store messages in session state so user can see chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Container to show conversation
    conversation_container = st.container()

    # Render existing conversation
    for msg in st.session_state["chat_history"]:
        if msg["role"]=="user":
            conversation_container.markdown(f"**You**: {msg['content']}")
        else:
            conversation_container.markdown(f"**Agent**: {msg['content']}")

    # Create container for new messages
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

        # Pass both containers to the callback handler
        cb_handler = StreamlitCallbackHandler(debug_container, conversation_container)
        agent = build_agent(callbacks=[cb_handler])

        with st.spinner("Thinking..."):
            try:
                ans_dict = agent.invoke({"input":user_msg})
                ans = ans_dict.get("output", "") if isinstance(ans_dict, dict) else ans_dict
                if ans.strip():
                    st.session_state["chat_history"].append({"role":"assistant","content":ans})
                    # Note: We don't need to display the answer here as the callback handler will do it
            except Exception as e:
                st.error(f"Agent error: {e}")

    st.text_input("Your message", key="user_input", on_change=submit_chat)

if __name__ == "__main__":
    main()