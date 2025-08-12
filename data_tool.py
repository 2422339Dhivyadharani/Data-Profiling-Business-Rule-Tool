import pandas as pd
import argparse
import json
import re


def profile_csv(file_path):
    df = pd.read_csv(file_path)
    profile = {
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'examples': {col: df[col].dropna().unique()[:3].tolist() for col in df.columns},
        'stats': df.describe(include='all').to_dict(),
        'row_count': len(df)
    }
    return profile

def prompt_for_rules(df):
    rules = {}
    print("\n--- Business Rule Setup ---")
    for col in df.columns:
        col_rules = {}
        dtype = df[col].dtype
        print(f"\nColumn: {col} (type: {dtype})")
        # Non-null
        non_null = input(f"Should '{col}' be non-null? (y/n): ").strip().lower() == 'y'
        if non_null:
            col_rules['non_null'] = True
        # Uniqueness
        unique = input(f"Should '{col}' be unique? (y/n): ").strip().lower() == 'y'
        if unique:
            col_rules['unique'] = True
        # Numeric rules
        if pd.api.types.is_numeric_dtype(dtype):
            min_val = input(f"Minimum value for '{col}' (blank for none): ").strip()
            max_val = input(f"Maximum value for '{col}' (blank for none): ").strip()
            if min_val:
                col_rules['min'] = float(min_val)
            if max_val:
                col_rules['max'] = float(max_val)
        # String rules
        if dtype == 'object':
            regex = input(f"Regex pattern for '{col}' (blank for none): ").strip()
            if regex:
                col_rules['regex'] = regex
        if col_rules:
            rules[col] = col_rules
    print("\nBusiness rules set:")
    print(json.dumps(rules, indent=2))
    return rules

def detect_errors(df, rules=None):
    errors = {}
    for col in df.columns:
        col_errors = []
        if df[col].isnull().any():
            col_errors.append('Has nulls')
        # Business rules
        if rules and col in rules:
            r = rules[col]
            if r.get('non_null') and df[col].isnull().any():
                col_errors.append('Nulls present (should be non-null)')
            if r.get('unique') and not df[col].is_unique:
                col_errors.append('Duplicates present (should be unique)')
            if 'min' in r and (df[col] < r['min']).any():
                col_errors.append(f'Below min {r["min"]}')
            if 'max' in r and (df[col] > r['max']).any():
                col_errors.append(f'Above max {r["max"]}')
            if 'regex' in r:
                pattern = re.compile(r['regex'])
                if df[col].dropna().apply(lambda x: not bool(pattern.match(str(x)))).any():
                    col_errors.append(f'Values do not match pattern {r["regex"]}')
        errors[col] = col_errors
    return errors

def suggest_corrections(df, errors, rules=None):
    corrections = {}
    for col, errs in errors.items():
        for err in errs:
            if 'null' in err.lower():
                corrections[col] = corrections.get(col, []) + ['Fill nulls with default', 'Drop rows with nulls']
            if 'duplicate' in err.lower():
                corrections[col] = corrections.get(col, []) + ['Drop duplicate rows']
            if 'below min' in err.lower():
                corrections[col] = corrections.get(col, []) + ['Set below min to min', 'Drop rows below min']
            if 'above max' in err.lower():
                corrections[col] = corrections.get(col, []) + ['Set above max to max', 'Drop rows above max']
            if 'pattern' in err.lower():
                corrections[col] = corrections.get(col, []) + ['Drop rows not matching pattern']
    return corrections

def apply_corrections(df, corrections, user_choices, rules=None):
    for col, actions in user_choices.items():
        for action in actions:
            if action == 'Fill nulls with default':
                if df[col].dtype == 'O':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 0)
            if action == 'Drop rows with nulls':
                df = df.dropna(subset=[col])
            if action == 'Drop duplicate rows':
                df = df.drop_duplicates(subset=[col])
            if action == 'Set below min to min' and rules and 'min' in rules.get(col, {}):
                df.loc[df[col] < rules[col]['min'], col] = rules[col]['min']
            if action == 'Drop rows below min' and rules and 'min' in rules.get(col, {}):
                df = df[df[col] >= rules[col]['min']]
            if action == 'Set above max to max' and rules and 'max' in rules.get(col, {}):
                df.loc[df[col] > rules[col]['max'], col] = rules[col]['max']
            if action == 'Drop rows above max' and rules and 'max' in rules.get(col, {}):
                df = df[df[col] <= rules[col]['max']]
            if action == 'Drop rows not matching pattern' and rules and 'regex' in rules.get(col, {}):
                pattern = re.compile(rules[col]['regex'])
                df = df[df[col].dropna().apply(lambda x: bool(pattern.match(str(x))))]
    return df

def main():
    parser = argparse.ArgumentParser(description='Data Profiling and Correction Tool')
    parser.add_argument('csv', help='Input CSV file')
    parser.add_argument('--output', help='Corrected CSV output file', default='corrected.csv')
    parser.add_argument('--auto', action='store_true', help='Apply all suggested corrections automatically')
    parser.add_argument('--rules', help='Path to JSON file with business rules (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    profile = profile_csv(args.csv)
    print('--- Profile ---')
    print(json.dumps(profile, indent=2))

    # Business rules: prompt user or load from JSON
    if args.rules:
        with open(args.rules) as f:
            rules = json.load(f)
        print('\nLoaded business rules from file:')
        print(json.dumps(rules, indent=2))
    else:
        rules = prompt_for_rules(df)

    errors = detect_errors(df, rules)
    print('\n--- Errors ---')
    print(json.dumps(errors, indent=2))

    corrections = suggest_corrections(df, errors, rules)
    print('\n--- Correction Suggestions ---')
    print(json.dumps(corrections, indent=2))

    if args.auto and corrections:
        user_choices = corrections
    elif corrections:
        user_choices = {}
        for col, actions in corrections.items():
            print(f'Column: {col}')
            for i, action in enumerate(actions):
                print(f'  [{i}] {action}')
            selected = input(f'Select actions for {col} (comma-separated indices, or blank for none): ')
            if selected:
                user_choices[col] = [actions[int(idx)] for idx in selected.split(',') if idx.isdigit() and int(idx) < len(actions)]
    else:
        user_choices = {}

    if user_choices:
        corrected = apply_corrections(df, corrections, user_choices, rules)
        corrected.to_csv(args.output, index=False)
        print(f'Corrected file written to {args.output}')
    else:
        print('No corrections applied.')

if __name__ == "__main__":
    main()
