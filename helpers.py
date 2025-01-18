import duckdb
import pandas as pd
import streamlit as st
from thefuzz import fuzz
from typing import Tuple, List, Set, Optional
import re

#############################
#   SANITIZE COLUMNS        #
#############################

def auto_sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to SQL-friendly forms:
      - Lowercase
      - Replace spaces & special chars with underscores
      - Ensure starts with letter
      - Remove invalid characters
      - Avoid SQL reserved words
    Returns the updated DataFrame.
    """
    new_cols = []
    for c in df.columns:
        # First pass: basic sanitization
        col = sanitize_sql_name(c)
        
        # If still not SQL-friendly, prefix with 'col_'
        if not is_sql_friendly_name(col):
            col = 'col_' + col
            
        # Handle duplicates by adding numbers
        base_col = col
        counter = 1
        while col in new_cols:
            col = f"{base_col}_{counter}"
            counter += 1
            
        new_cols.append(col)
    
    df.columns = new_cols
    return df

#############################
#   SCHEMA & DATA INSPECTION
#############################

def inspect_schema(conn: duckdb.DuckDBPyConnection, table_name: str) -> List[Tuple[str, str]]:
    try:
        schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
        return [(col[0], col[1]) for col in schema_info]
    except Exception as e:
        st.error(f"Error inspecting schema for {table_name}: {e}")
        return []

def get_schema_description(conn: duckdb.DuckDBPyConnection, tables: List[str]) -> str:
    if not tables:
        return "No tables found in database."

    lines = []
    for tbl in tables:
        cols = inspect_schema(conn, tbl)
        if cols:
            lines.append(f"Table: {tbl}")
            for cname, ctype in cols:
                lines.append(f"  - {cname} ({ctype})")
    if not lines:
        return "No tables found in database."
    return "\n".join(lines)

def get_all_columns(conn: duckdb.DuckDBPyConnection, tables: List[str]) -> Set[str]:
    """
    Return a set of all column names (lowercased) across the given tables.
    """
    all_cols = set()
    for tbl in tables:
        try:
            schema_data = conn.execute(f'DESCRIBE "{tbl}"').fetchall()
            for row in schema_data:
                col_name = row[0].lower()  # row[0] is the column name
                all_cols.add(col_name)
        except Exception:
            # If table doesn't exist or some error, skip
            pass
    return all_cols

def is_sql_friendly_name(name: str) -> bool:
    """
    Check if a column name is SQL-friendly:
    - Starts with a letter
    - Contains only letters, numbers, and underscores
    - Not a SQL reserved word
    """
    if not name:
        return False
    
    # Must start with letter
    if not name[0].isalpha():
        return False
    
    # Only allow letters, numbers, underscores
    if not all(c.isalnum() or c == '_' for c in name):
        return False
    
    # Common SQL reserved words to avoid
    sql_reserved = {
        'select', 'from', 'where', 'group', 'order', 'by', 'having',
        'join', 'inner', 'outer', 'left', 'right', 'on', 'as', 'case',
        'when', 'then', 'else', 'end', 'and', 'or', 'not', 'null',
        'true', 'false', 'like', 'in', 'between', 'is', 'exists'
    }
    
    return name.lower() not in sql_reserved

def sanitize_sql_name(name: str) -> str:
    """
    Convert a name to SQL-friendly format:
    - Replace spaces and special chars with underscores
    - Ensure starts with letter
    - Remove any other invalid characters
    """
    # Convert to lowercase and replace spaces/special chars with underscore
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
    
    # Ensure starts with letter
    if not clean[0].isalpha():
        clean = 'col_' + clean
    
    # Remove duplicate underscores
    clean = re.sub(r'_+', '_', clean)
    
    # Trim underscores from ends
    clean = clean.strip('_')
    
    return clean

#############################
#   SQL VALIDATION & FUZZY
#############################

def validate_sql_query(conn: duckdb.DuckDBPyConnection, sql_query: str, tables: List[str]) -> Tuple[bool, str]:
    try:
        conn.execute(f"EXPLAIN {sql_query}")
    except Exception as e:
        return False, f"Validation error (syntax): {str(e)}"
    return True, "SQL query validated successfully."

def fuzzy_match_column(user_input: str, columns: List[str], threshold: int = 70) -> str:
    best = None
    best_score = 0
    for col in columns:
        score = fuzz.ratio(user_input.lower(), col.lower())
        if score > best_score:
            best_score = score
            best = col
    return best if best_score > threshold else user_input

# helpers.py

def apply_fuzzy_matching_to_query(
    sql_query: str,
    table_names: List[str],
    conn: duckdb.DuckDBPyConnection
) -> str:
    """
    We only allow one table at a time, so we ignore any table reference 
    from the user and force references to that single known table. 
    We only fuzzy-match columns.
    """
    # If there's exactly one table loaded, let's get it:
    if len(table_names) == 1:
        single_table = table_names[0]
    else:
        # Fallback if somehow multiple tables exist
        return sql_query  # do nothing special
    
    # Get all columns for that single table
    all_cols = get_all_columns(conn, [single_table])  # your existing helper

    import re
    tokens = re.split(r'(\s+|\W+)', sql_query)  # split on whitespace or punctuation

    revised_tokens = []
    for tok in tokens:
        raw_tok = tok.strip('"`[] ')

        # If the token is the user-typed table name, forcibly replace it 
        # with our single_table, ignoring any user-typed reference:
        # (or if the user typed anything that looks like a table name, we ignore it)
        if raw_tok.lower() == single_table.lower():
            revised_tokens.append(f'"{single_table}"')
            continue

        # Check if the token matches a column (exact)
        if raw_tok in all_cols:
            revised_tokens.append(tok)  # keep as is
            continue
        
        # Attempt fuzzy match on columns only
        matched_col = fuzzy_match_column(raw_tok, list(all_cols), threshold=80)
        if matched_col != raw_tok:
            revised_tokens.append(f'"{matched_col}"')
        else:
            revised_tokens.append(tok)

    # Now forcibly replace any FROM clauses with the single_table:
    # e.g. if user wrote "FROM some_fake_name", override to single_table.
    # This is optional if you want to guarantee the user can’t override the table name:
    revised_query = "".join(revised_tokens)
    revised_query = re.sub(r'(?i)\bFROM\s+["`]?[\w]+["`]?\b', f'FROM "{single_table}"', revised_query)

    return revised_query

#############################
#   DATA LOADING
#############################

def create_table_from_csv(conn: duckdb.DuckDBPyConnection, csv_file: str, table_name: str) -> Optional[List[Tuple[str, str]]]:
    try:
        df = pd.read_csv(csv_file)
        # SANITIZE columns
        df = auto_sanitize_columns(df)

        conn.execute(f"DROP VIEW IF EXISTS {table_name}")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.register(table_name, df)
        return inspect_schema(conn, table_name)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def create_table_from_excel(conn: duckdb.DuckDBPyConnection, excel_file: str, table_name: str) -> Optional[List[Tuple[str, str]]]:
    try:
        xls = pd.ExcelFile(excel_file)
        sheet = xls.sheet_names[0]
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox(f"Select a sheet for {table_name}:", xls.sheet_names)
        df = pd.read_excel(excel_file, sheet_name=sheet)
        # SANITIZE columns
        df = auto_sanitize_columns(df)

        conn.execute(f"DROP VIEW IF EXISTS {table_name}")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.register(table_name, df)
        return inspect_schema(conn, table_name)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

#############################
#   QUERY EXECUTION
#############################

def execute_sql_and_return_results(conn: duckdb.DuckDBPyConnection, sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = conn.execute(sql_query).fetchdf()
        return df, None
    except Exception as e:
        return None, str(e)

#############################
#   MULTI-STRATEGY CLEANING
#############################

def data_cleaning_tool(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    strategy: str = "numeric_only"
) -> str:
    if table_name not in st.session_state["loaded_tables"]:
        return f"Table '{table_name}' not found."

    try:
        df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found in '{table_name}'"

        original_len = len(df)
        
        if strategy == "numeric_only":
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            df = df.dropna(subset=[column_name])
            dropped = original_len - len(df)

            conn.execute(f"DROP VIEW IF EXISTS {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.register(table_name, df)
            return f"Cleaned '{column_name}' in '{table_name}', removed {dropped} non-numeric rows."

        elif strategy == "fill_zero":
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
            conn.execute(f"DROP VIEW IF EXISTS {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.register(table_name, df)
            return f"Cleaned '{column_name}' in '{table_name}', replaced non-numeric with 0."

        elif strategy == "strip_whitespace":
            if df[column_name].dtype == object:
                df[column_name] = df[column_name].astype(str).str.strip()
                conn.execute(f"DROP VIEW IF EXISTS {table_name}")
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.register(table_name, df)
                return f"Stripped whitespace in '{column_name}' of '{table_name}'."
            else:
                return f"Column '{column_name}' is not string-based, can't strip whitespace."
        
        return f"Unknown strategy: {strategy}"

    except Exception as e:
        return f"Data cleaning error: {e}"

#############################
#   SEMANTIC COLUMN MAPPING
#############################

def semantic_column_selection(
    conn: duckdb.DuckDBPyConnection,
    user_question: str,
    table_names: List[str]
) -> List[str]:
    # If no data loaded, none
    if not table_names:
        return []

    # If exactly 1 table (excluding demo_data?), auto fallback
    filtered = [t for t in table_names if t!="demo_data"]
    if len(filtered)==1:
        one_tbl = filtered[0]
        schema_info = inspect_schema(conn, one_tbl)
        matched_cols = []
        qlower = user_question.lower()
        for cname, _ct in schema_info:
            if cname in qlower:
                matched_cols.append(f"{one_tbl}.{cname}")
        if not matched_cols:
            # fallback: all columns from that single table
            matched_cols = [f"{one_tbl}.{c[0]}" for c in schema_info]
        return matched_cols

    # multiple tables
    matched = []
    qlower = user_question.lower()
    for tbl in filtered:
        schema_info = inspect_schema(conn, tbl)
        for cname,_ct in schema_info:
            if cname in qlower:
                matched.append(f"{tbl}.{cname}")
    return matched

def semantic_col_selector_tool(args: str) -> str:
    """
    Return potential columns from loaded tables that might be relevant.
    If no exact match, just provide all columns from relevant tables so
    the user or agent can refine.
    """
    conn = st.session_state["duckdb_conn"]
    loaded_tables = st.session_state.get("loaded_tables", [])
    if not loaded_tables:
        return "No tables loaded."

    # If the user passes something like user_question='Which region has the most products sold?'
    # you can optionally parse out keywords, then check columns. For now, let's simplify:
    user_q = args.strip()
    if "=" in args:  # might be "user_question=some question"
        data = {}
        for p in args.split(","):
            if "=" in p:
                k, v = p.split("=",1)
                data[k.strip()] = v.strip()
        user_q = data.get("user_question", user_q)

    if not user_q:
        return "No question provided."

    # For each table, gather columns
    all_suggestions = []
    for t in loaded_tables:
        col_list = get_all_columns(conn, [t])
        all_suggestions.extend( [f"{t}.{c}" for c in col_list] )

    # Return them to the LLM so it can pick any that might be relevant
    if all_suggestions:
        return f"Potential columns: {', '.join(all_suggestions)}"
    else:
        return "No columns found in loaded tables."