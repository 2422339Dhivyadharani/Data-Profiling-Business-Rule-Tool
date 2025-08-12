# Data Profiling Business Rule Tool

This project is a simple Extract-Transform-Load (ETL) pipeline using Python and CSV files.

## How it works
- **Extract:** Reads data from `data/data_profiling_business_rule_tool.csv`
- **Transform:** Filters out people under 30 years old
- **Load:** Writes the filtered data to `data/output.csv`

## Running the ETL
```bash
python data_profiling_business_rule_tool/etl.py
```

## Running Tests
```bash
pytest data_profiling_business_rule_tool/tests/
```
