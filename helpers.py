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
    Convert column names to simpler forms:
      - Lowercase
      - Replace spaces & parentheses with underscores
      - Remove special chars
    Returns the updated DataFrame.
    """
    new_cols = []
    for c in df.columns:
        # Lowercase
        col = c.lower()
        # Replace parentheses or spaces with underscores
        col = re.sub(r"[()\s]+", "_", col)
        # Remove any other non-alphanumeric except underscores
        col = re.sub(r"[^a-z0-9_]+", "", col)
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
    all_cols = set()
    for tbl in tables:
        schema_data = inspect_schema(conn, tbl)
        for cname, _ctype in schema_data:
            all_cols.add(cname)
    return all_cols

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

def apply_fuzzy_matching_to_query(sql_query: str, table_names: List[str], conn: duckdb.DuckDBPyConnection) -> str:
    # valid_tables = [t for t in table_names if t != "demo_data"]
    valid_tables = table_names 
    all_cols = get_all_columns(conn, valid_tables)

    tokens = sql_query.replace(",", " , ").replace("(", " ( ").replace(")", " ) ").split()
    revised_tokens = []
    for tok in tokens:
        raw = tok.strip('"[]\'')
        if raw in all_cols:
            revised_tokens.append(tok)
        else:
            matched = fuzzy_match_column(raw, list(all_cols))
            if matched != raw:
                revised_tokens.append(f'"{matched}"')
            else:
                revised_tokens.append(tok)
    return " ".join(revised_tokens)

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