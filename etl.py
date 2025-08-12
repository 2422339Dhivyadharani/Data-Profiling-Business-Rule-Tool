import csv
import sqlite3
import os

def extract(data_profiling_business_rule_tool_file):
    with open(data_profiling_business_rule_tool_file, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def transform(data):
    # Remove rows with any null (empty string) values
    cleaned = [row for row in data if all(row.values())]
    # Add a 'salary' column (for demo: salary = age * 1000)
    for row in cleaned:
        row['salary'] = str(int(row['age']) * 1000)
    return cleaned

def create_db_schema(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            salary INTEGER NOT NULL
        );
    ''')
    conn.commit()

def load_to_sqlite(data, db_path):
    if not data:
        return
    conn = sqlite3.connect(db_path)
    create_db_schema(conn)
    # Clear table for idempotency
    conn.execute('DELETE FROM employees;')
    # Insert rows
    for row in data:
        conn.execute(
            'INSERT INTO employees (id, name, age, salary) VALUES (?, ?, ?, ?)',
            (int(row['id']), row['name'], int(row['age']), int(row['salary']))
        )
    conn.commit()
    conn.close()

def etl(data_profiling_business_rule_tool_file, db_path):
    data = extract(data_profiling_business_rule_tool_file)
    transformed = transform(data)
    load_to_sqlite(transformed, db_path)

if __name__ == "__main__":
    etl('data_profiling_business_rule_tool/data/data_profiling_business_rule_tool.csv', 'data_profiling_business_rule_tool/data/etl_demo.sqlite')
