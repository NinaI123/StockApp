import sqlite3

DB_FILE = "data/portfolio.db"

def print_table(table_name: str):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    print(f"\n--- {table_name.upper()} ---")
    for row in rows:
        print(dict(row))

    conn.close()


if __name__ == "__main__":
    tables = [
        "trades",
        "watchlist",
        "pw_leagues",
        "pw_teams",
        "pw_matchups",
        "pw_results"
    ]

    for t in tables:
        print_table(t)
