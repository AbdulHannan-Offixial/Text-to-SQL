import sqlite3

def extract_database_schema(db_path: str) -> str:
    """Connects to SQLite and returns a clean, formatted schema string."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch all table names, ignoring system tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
    
    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        columns = cursor.fetchall()
        # col[1] = column name, col[2] = data type
        col_descriptions = [f"{col[1]} ({col[2]})" for col in columns]
        schema_parts.append(f"Table: {table}\nColumns: {', '.join(col_descriptions)}")
    
    conn.close()
    return "\n\n".join(schema_parts)

# Quick local sanity check
if __name__ == "__main__":
    # Create a temporary in-memory database to test extraction
    test_conn = sqlite3.connect("sample.db")
    test_conn.execute("CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT, total_spend REAL);")
    test_conn.execute("CREATE TABLE IF NOT EXISTS orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);")
    test_conn.close()
    
    formatted_schema = extract_database_schema("sample.db")
    print("Extracted Schema Output:\n")
    print(formatted_schema)