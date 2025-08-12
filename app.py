import streamlit as st
import pandas as pd
import json
import re
from io import StringIO

def profile_csv(df):
    return {
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'examples': {col: df[col].dropna().unique()[:3].tolist() for col in df.columns},
        'stats': df.describe(include='all').to_dict(),
        'row_count': len(df)
    }

def visualize_data(df, rules=None, errors=None):
    st.subheader('Column Charts')
    # Let user select columns to plot
    all_cols = list(df.columns)
    low_card_cols = [col for col in all_cols if df[col].nunique() < 20 or pd.api.types.is_numeric_dtype(df[col])]
    selected_cols = st.multiselect(
        'Select columns to visualize (categorical with <20 unique values and all numerics are pre-selected)',
        all_cols,
        default=low_card_cols,
        key='visualize_cols_main'
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    for col in selected_cols:
        st.markdown(f'**{col}**')
        # Sample size option for large columns
        data = df[col].dropna()
        sample_size = None
        if len(data) > 1000:
            sample_size = st.number_input(f'Sample size for {col} (max {len(data)})', min_value=100, max_value=len(data), value=1000, step=100, key=f'{col}_sample')
            data = data.sample(n=int(sample_size), random_state=42) if sample_size < len(data) else data
            st.write(f'Displaying a sample of {len(data)} rows.')
        # Let user pick chart type
        chart_type = st.selectbox(
            f'Chart type for {col}',
            ['Bar', 'Histogram', 'Boxplot', 'Pie'] if pd.api.types.is_numeric_dtype(data) or data.nunique() < 20 else ['Bar'],
            key=f'{col}_charttype'
        )
        if chart_type == 'Bar':
            if pd.api.types.is_numeric_dtype(data):
                st.bar_chart(data)
            else:
                st.bar_chart(data.value_counts())
        elif chart_type == 'Histogram' and pd.api.types.is_numeric_dtype(data):
            fig, ax = plt.subplots()
            sns.histplot(data, kde=True, ax=ax)
            st.pyplot(fig)
        elif chart_type == 'Boxplot' and pd.api.types.is_numeric_dtype(data):
            fig, ax = plt.subplots()
            sns.boxplot(x=data, ax=ax)
            st.pyplot(fig)
        elif chart_type == 'Pie' and data.nunique() < 20:
            fig, ax = plt.subplots()
            data.value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)
        else:
            st.write(f"Chart type '{chart_type}' is not available for this column.")
    if errors:
        st.subheader('Rule Violations (Counts)')
        err_counts = {col: len(val) for col, val in errors.items() if val}
        if err_counts:
            st.bar_chart(err_counts)
        else:
            st.write('No rule violations!')

def detect_errors(df, rules=None):
    errors = {}
    for col in df.columns:
        col_errors = []
        if df[col].isnull().any():
            col_errors.append('Has nulls')
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

def apply_corrections(df, rules):
    # For demo: just drop rows violating non_null, min, max, regex, unique
    for col, r in rules.items():
        if r.get('non_null'):
            df = df.dropna(subset=[col])
        if r.get('unique'):
            df = df.drop_duplicates(subset=[col])
        if 'min' in r:
            df = df[df[col] >= r['min']]
        if 'max' in r:
            df = df[df[col] <= r['max']]
        if 'regex' in r:
            pattern = re.compile(r['regex'])
            df = df[df[col].dropna().apply(lambda x: bool(pattern.match(str(x))))]
    return df

def generate_pytest_code(rules):
    code = ["import pandas as pd\n\ndef test_business_rules():\n    df = pd.read_csv('your_file.csv')"]
    for col, r in rules.items():
        if r.get('non_null'):
            code.append(f"    assert df['{col}'].isnull().sum() == 0  # {col} should be non-null")
        if r.get('unique'):
            code.append(f"    assert df['{col}'].is_unique  # {col} should be unique")
        if 'min' in r:
            code.append(f"    assert (df['{col}'] >= {r['min']}).all()  # {col} >= {r['min']}")
        if 'max' in r:
            code.append(f"    assert (df['{col}'] <= {r['max']}).all()  # {col} <= {r['max']}")
        if 'regex' in r:
            code.append(f"    import re\n    pattern = re.compile(r'{r['regex']}')\n    assert df['{col}'].dropna().apply(lambda x: bool(pattern.match(str(x)))).all()  # {col} matches regex")
    return '\n'.join(code)

def main():
    st.title('Data Profiling & Business Rule Tool')
    st.write('Upload a CSV, define business rules, validate, correct, and generate test code!')

    uploaded_file = st.file_uploader('Upload CSV', type='csv')
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader('Data Preview')
        st.dataframe(df.head())
        profile = profile_csv(df)
        # st.subheader('Profile Table')
        # Build profile table
        profile_table = pd.DataFrame({
            'dtype': profile['dtypes'],
            'null_count': profile['null_counts'],
            'unique_count': profile['unique_counts'],
            'examples': {k: ', '.join(map(str, v)) for k, v in profile['examples'].items()},
        })
        # Optionally, merge in stats if present
        stats_df = pd.DataFrame(profile['stats']) if 'stats' in profile else None
        if stats_df is not None and not stats_df.empty:
            for stat_row in stats_df.index:
                profile_table[stat_row] = stats_df.loc[stat_row]
        st.subheader('Profile Table')
        st.markdown('**Columns:** ' + ', '.join(profile_table.columns))
        st.dataframe(profile_table)
        # Add download button for profile table
        csv_profile = profile_table.to_csv().encode('utf-8')
        st.download_button('Download Profile Table as CSV', csv_profile, 'profile_table.csv', 'text/csv')

        st.markdown('---')
        st.subheader('Data Visualization')
        show_charts = st.checkbox('Show data visualizations', value=False)
        if show_charts:
            visualize_data(df)

        st.markdown('---')
        st.subheader('Advanced Profiling')
        adv_outlier = st.checkbox('Outlier Detection (IQR/Z-score)', value=False)
        adv_quality = st.checkbox('Data Quality Metrics', value=False)
        adv_pattern = st.checkbox('Pattern Recognition', value=False)
        adv_corr = st.checkbox('Correlation Analysis', value=False)
        adv_drill = st.checkbox('Drill-down Profiling', value=False)

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        if adv_outlier:
            st.markdown('**Outlier Detection (IQR/Z-score)**')
            outlier_table = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                data = df[col].dropna()
                # IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                iqr_out = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                # Z-score
                zscores = (data - data.mean()) / data.std(ddof=0)
                z_out = data[np.abs(zscores) > 3]
                outlier_table[col] = {
                    'IQR outlier count': len(iqr_out),
                    'Z-score outlier count': len(z_out),
                    'IQR outlier %': 100*len(iqr_out)/len(data) if len(data) else 0,
                    'Z-score outlier %': 100*len(z_out)/len(data) if len(data) else 0
                }
            st.dataframe(pd.DataFrame(outlier_table).T)

        if adv_quality:
            st.markdown('**Data Quality Metrics**')
            metrics = {}
            for col in df.columns:
                total = len(df)
                non_null = df[col].notnull().sum()
                unique = df[col].nunique(dropna=True)
                # Consistency: percent values matching allowed_values or regex (if rule exists)
                consistency = np.nan
                validity = np.nan
                if 'rules' in st.session_state and col in st.session_state['rules']:
                    rule = st.session_state['rules'][col]
                    if 'allowed_values' in rule:
                        allowed = set(rule['allowed_values'])
                        consistency = 100 * df[col].isin(allowed).mean()
                    if 'regex' in rule:
                        pattern = rule['regex']
                        consistency = 100 * df[col].astype(str).str.match(pattern).mean()
                    # Validity: for numerics, within min/max; for strings, regex
                    if pd.api.types.is_numeric_dtype(df[col]):
                        v = pd.Series([True]*total)
                        if 'min' in rule:
                            v &= df[col] >= rule['min']
                        if 'max' in rule:
                            v &= df[col] <= rule['max']
                        validity = 100 * v.mean()
                    elif 'regex' in rule:
                        validity = 100 * df[col].astype(str).str.match(rule['regex']).mean()
                metrics[col] = {
                    'Completeness %': 100*non_null/total if total else 0,
                    'Uniqueness %': 100*unique/total if total else 0,
                    'Consistency %': consistency if not np.isnan(consistency) else '',
                    'Validity %': validity if not np.isnan(validity) else '',
                    'Accuracy % (placeholder)': ''
                }
            st.dataframe(pd.DataFrame(metrics).T)

        if adv_pattern:
            st.markdown('**Pattern Recognition**')
            import re
            pattern_table = {}
            for col in df.select_dtypes(include='object').columns:
                patterns = df[col].dropna().astype(str).str.extract(r'([\w\-@\.]+)')
                most_common = patterns[0].value_counts().head(5)
                pattern_table[col] = {f'Pattern {i+1}': v for i, v in enumerate(most_common.index)}
            st.dataframe(pd.DataFrame(pattern_table))

        if adv_corr:
            st.markdown('**Correlation Analysis**')
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(min(8, len(num_cols)), min(6, len(num_cols))))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                # Scatter plot
                xcol = st.selectbox('X for scatter', num_cols, key='corr_x')
                ycol = st.selectbox('Y for scatter', num_cols, key='corr_y')
                if xcol != ycol:
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(df[xcol], df[ycol], alpha=0.4)
                    ax2.set_xlabel(xcol)
                    ax2.set_ylabel(ycol)
                    st.pyplot(fig2)
            else:
                st.write('Not enough numeric columns for correlation analysis.')

        if adv_drill:
            st.markdown('**Drill-down Profiling**')
            group_col = st.selectbox('Select column to group by (for drill-down):', df.columns, key='drill_group')
            group_val = st.selectbox('Select value to drill into:', sorted(df[group_col].dropna().unique()), key='drill_val')
            sub_df = df[df[group_col] == group_val]
            st.write(f'Showing profile for {group_col} = {group_val} (n={len(sub_df)})')
            sub_profile = profile_csv(sub_df)
            sub_profile_table = pd.DataFrame({
                'dtype': sub_profile['dtypes'],
                'null_count': sub_profile['null_counts'],
                'unique_count': sub_profile['unique_counts'],
                'examples': {k: ', '.join(map(str, v)) for k, v in sub_profile['examples'].items()},
            })
            st.dataframe(sub_profile_table)

        st.subheader('Business Rule Builder')
        # Use session state to persist rules
        if 'rules' not in st.session_state:
            st.session_state['rules'] = {}
        rules = st.session_state['rules']
        # Dropdown to select column
        col_to_edit = st.selectbox('Select a column to define business rules:', df.columns)
        col_rules = {}
        st.markdown(f'**Column: {col_to_edit} ({df[col_to_edit].dtype})**')
        col_rules['non_null'] = st.checkbox(f'Non-null', key=f'{col_to_edit}_non_null')
        col_rules['unique'] = st.checkbox(f'Unique', key=f'{col_to_edit}_unique')
        if pd.api.types.is_numeric_dtype(df[col_to_edit]):
            min_val = st.text_input('Min value', key=f'{col_to_edit}_min')
            max_val = st.text_input('Max value', key=f'{col_to_edit}_max')
            if min_val:
                try: col_rules['min'] = float(min_val)
                except: pass
            if max_val:
                try: col_rules['max'] = float(max_val)
                except: pass
        if df[col_to_edit].dtype == 'object':
            regex = st.text_input('Regex pattern', key=f'{col_to_edit}_regex')
            if regex:
                col_rules['regex'] = regex
            allowed_values = st.text_input('Allowed values (comma-separated)', key=f'{col_to_edit}_allowed')
            if allowed_values:
                col_rules['allowed_values'] = [v.strip() for v in allowed_values.split(',') if v.strip()]
        if pd.api.types.is_datetime64_any_dtype(df[col_to_edit]):
            min_date = st.date_input('Min date', key=f'{col_to_edit}_min_date')
            max_date = st.date_input('Max date', key=f'{col_to_edit}_max_date')
            col_rules['min_date'] = min_date
            col_rules['max_date'] = max_date
        custom_expr = st.text_input('Custom validation (Python, e.g. x > 0)', key=f'{col_to_edit}_expr')
        if custom_expr:
            col_rules['custom_expr'] = custom_expr
        # Add/Update rules for this column
        if st.button('Save/Update Rules for this Column'):
            rules[col_to_edit] = {k: v for k, v in col_rules.items() if v}
            st.success(f'Rules for {col_to_edit} saved!')
        # Show rules set so far
        if rules:
            st.markdown('**Business Rules Defined:**')
            st.json(rules)

        # Data Transformation & Cleansing Section
        st.markdown('---')
        st.subheader('Data Transformation & Cleansing')
        # Use session state for transformed data
        if 'transformed_df' not in st.session_state:
            st.session_state['transformed_df'] = df.copy()
        tdf = st.session_state['transformed_df']
        col_to_clean = st.selectbox('Select a column for cleaning/transformation:', tdf.columns, key='clean_col')
        st.write(f'Preview: {col_to_clean}', tdf[col_to_clean].head(10))
        # Suggest fixes
        missing = tdf[col_to_clean].isnull().sum()
        dups = tdf[col_to_clean].duplicated().sum()
        st.markdown(f'- Missing values: {missing}')
        st.markdown(f'- Duplicates: {dups}')
        # Suggest actions
        fix_action = st.selectbox('Suggested fix:', ['None','Fill missing (mean/Unknown)','Drop missing','Drop duplicates','Standardize format','Custom regex','Custom formula'])
        if fix_action == 'Fill missing (mean/Unknown)':
            if pd.api.types.is_numeric_dtype(tdf[col_to_clean]):
                if st.button('Fill missing with mean'):
                    tdf[col_to_clean] = tdf[col_to_clean].fillna(tdf[col_to_clean].mean())
                    st.success('Filled missing values with mean.')
            else:
                if st.button('Fill missing with "Unknown"'):
                    tdf[col_to_clean] = tdf[col_to_clean].fillna('Unknown')
                    st.success('Filled missing values with "Unknown".')
        elif fix_action == 'Drop missing':
            if st.button('Drop missing values'):
                tdf = tdf.dropna(subset=[col_to_clean])
                st.session_state['transformed_df'] = tdf
                st.success('Dropped missing values.')
        elif fix_action == 'Drop duplicates':
            if st.button('Drop duplicates'):
                tdf = tdf.drop_duplicates(subset=[col_to_clean])
                st.session_state['transformed_df'] = tdf
                st.success('Dropped duplicates.')
        elif fix_action == 'Standardize format':
            if pd.api.types.is_datetime64_any_dtype(tdf[col_to_clean]):
                if st.button('Standardize date format (YYYY-MM-DD)'):
                    tdf[col_to_clean] = pd.to_datetime(tdf[col_to_clean], errors='coerce').dt.strftime('%Y-%m-%d')
                    st.session_state['transformed_df'] = tdf
                    st.success('Date format standardized.')
            elif pd.api.types.is_numeric_dtype(tdf[col_to_clean]):
                if st.button('Standardize currency (remove symbols)'):
                    tdf[col_to_clean] = tdf[col_to_clean].replace(r'[^\d.]', '', regex=True).astype(float)
                    st.session_state['transformed_df'] = tdf
                    st.success('Currency standardized.')
            else:
                if st.button('Standardize names (title case, strip)'):
                    tdf[col_to_clean] = tdf[col_to_clean].astype(str).str.strip().str.title()
                    st.session_state['transformed_df'] = tdf
                    st.success('Names standardized.')
        elif fix_action == 'Custom regex':
            regex_pat = st.text_input('Regex pattern to extract/replace (e.g., [A-Za-z]+):', key='regex_pat')
            regex_repl = st.text_input('Replacement (optional):', key='regex_repl')
            if st.button('Apply regex'):
                if regex_repl:
                    tdf[col_to_clean] = tdf[col_to_clean].astype(str).str.replace(regex_pat, regex_repl, regex=True)
                else:
                    tdf[col_to_clean] = tdf[col_to_clean].astype(str).str.extract(regex_pat, expand=False)
                st.session_state['transformed_df'] = tdf
                st.success('Regex transformation applied.')
        elif fix_action == 'Custom formula':
            formula = st.text_input('Enter formula (use x for value, e.g., x*2, x.upper()):', key='formula')
            if st.button('Apply formula'):
                try:
                    tdf[col_to_clean] = tdf[col_to_clean].apply(lambda x: eval(formula, {"x": x, "np": np, "pd": pd}))
                    st.session_state['transformed_df'] = tdf
                    st.success('Formula applied.')
                except Exception as e:
                    st.error(f'Error applying formula: {e}')
        # Show preview and download
        st.write('Transformed Preview:', tdf.head(10))
        tdf_csv = tdf.to_csv(index=False).encode('utf-8')
        st.download_button('Download Transformed Data', tdf_csv, 'transformed_data.csv', 'text/csv')

        # Visualization & Reporting Section
        st.markdown('---')
        st.subheader('Visualization & Reporting')
        # Widget selection for dashboard
        widgets = st.multiselect('Select widgets for dashboard:', ['Charts','Data Quality Metrics','Outlier Table','Pattern Table','Correlation Heatmap'], default=['Charts'], key='dashboard_widgets')
        if 'Charts' in widgets:
            show_charts = st.checkbox('Show dashboard charts', value=False, key='dashboard_charts')
            if show_charts:
                visualize_data(df)
        import numpy as np
        import re
        # Data Quality Metrics
        if 'Data Quality Metrics' in widgets:
            if 'metrics' not in st.session_state:
                metrics = {}
                for col in df.columns:
                    total = len(df)
                    non_null = df[col].notnull().sum()
                    unique = df[col].nunique(dropna=True)
                    consistency = np.nan
                    validity = np.nan
                    if 'rules' in st.session_state and col in st.session_state['rules']:
                        rule = st.session_state['rules'][col]
                        if 'allowed_values' in rule:
                            allowed = set(rule['allowed_values'])
                            consistency = 100 * df[col].isin(allowed).mean()
                        if 'regex' in rule:
                            pattern = rule['regex']
                            consistency = 100 * df[col].astype(str).str.match(pattern).mean()
                        if pd.api.types.is_numeric_dtype(df[col]):
                            v = pd.Series([True]*total)
                            if 'min' in rule:
                                v &= df[col] >= rule['min']
                            if 'max' in rule:
                                v &= df[col] <= rule['max']
                            validity = 100 * v.mean()
                        elif 'regex' in rule:
                            validity = 100 * df[col].astype(str).str.match(rule['regex']).mean()
                    metrics[col] = {
                        'Completeness %': 100*non_null/total if total else 0,
                        'Uniqueness %': 100*unique/total if total else 0,
                        'Consistency %': consistency if not np.isnan(consistency) else '',
                        'Validity %': validity if not np.isnan(validity) else '',
                        'Accuracy % (placeholder)': ''
                    }
                st.session_state['metrics'] = metrics
            st.write('Data Quality Metrics')
            st.dataframe(pd.DataFrame(st.session_state['metrics']).T)
        # Outlier Table
        if 'Outlier Table' in widgets:
            if 'outlier_table' not in st.session_state:
                outlier_table = {}
                for col in df.select_dtypes(include=[np.number]).columns:
                    data = df[col].dropna()
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    iqr_out = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                    zscores = (data - data.mean()) / data.std(ddof=0)
                    z_out = data[np.abs(zscores) > 3]
                    outlier_table[col] = {
                        'IQR outlier count': len(iqr_out),
                        'Z-score outlier count': len(z_out),
                        'IQR outlier %': 100*len(iqr_out)/len(data) if len(data) else 0,
                        'Z-score outlier %': 100*len(z_out)/len(data) if len(data) else 0
                    }
                st.session_state['outlier_table'] = outlier_table
            st.write('Outlier Table')
            st.dataframe(pd.DataFrame(st.session_state['outlier_table']).T)
        # Pattern Table
        if 'Pattern Table' in widgets:
            if 'pattern_table' not in st.session_state:
                pattern_table = {}
                for col in df.select_dtypes(include='object').columns:
                    patterns = df[col].dropna().astype(str).str.extract(r'([\w\-@\.]+)')
                    most_common = patterns[0].value_counts().head(5)
                    pattern_table[col] = {f'Pattern {i+1}': v for i, v in enumerate(most_common.index)}
                st.session_state['pattern_table'] = pattern_table
            st.write('Pattern Table')
            st.dataframe(pd.DataFrame(st.session_state['pattern_table']))
        # Correlation Heatmap
        if 'Correlation Heatmap' in widgets:
            if 'corr' not in st.session_state:
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    st.session_state['corr'] = corr
                else:
                    st.session_state['corr'] = None
            corr = st.session_state['corr']
            if corr is not None:
                st.write('Correlation Heatmap')
                fig, ax = plt.subplots(figsize=(min(8, len(corr)), min(6, len(corr))))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.write('Not enough numeric columns for correlation analysis.')
        # Anomaly Chart Example
        st.markdown('**Anomaly Chart Example (Z-score > 3)**')
        anomaly_found = False
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) < 3 or data.std(ddof=0) == 0:
                continue  # Not enough data or no variation
            zscores = (data - data.mean()) / data.std(ddof=0)
            anomalies_idx = data.index[abs(zscores) > 3]
            if len(anomalies_idx) > 0:
                st.write(f'Anomalies in {col}:')
                st.dataframe(df.loc[anomalies_idx, [col]].head(10))
                anomaly_found = True
        if not anomaly_found:
            st.info('No anomalies detected (Z-score > 3) in numeric columns.')
        # Export Options
        st.markdown('---')
        st.subheader('Export Reports')
        # Show summary in app
        summary_data = {
            'Total Rows': len(df),
            'Total Columns': len(df.columns),
            'Total Nulls': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
        }
        notes = []
        if 'outlier_table' in st.session_state:
            notes.append('Outlier columns: ' + ', '.join([col for col, vals in st.session_state['outlier_table'].items() if vals['IQR outlier count'] or vals['Z-score outlier count']]))
        if 'metrics' in st.session_state:
            completeness = pd.DataFrame(st.session_state['metrics']).T['Completeness %'].mean()
            notes.append(f'Average completeness: {completeness:.2f}%')
        st.markdown('### Overall Summary')
        summary_lines = [
            f"- Your dataset has **{summary_data['Total Rows']} rows** and **{summary_data['Total Columns']} columns**.",
            f"- There are **{summary_data['Total Nulls']} missing values** and **{summary_data['Duplicate Rows']} duplicate rows** detected."
        ]
        if notes:
            for note in notes:
                summary_lines.append(f"- {note}")
        st.markdown('\n'.join(summary_lines))
        import io
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Raw Data')
            profile_table.to_excel(writer, sheet_name='Profile Table')
            # Data Quality Metrics
            if 'metrics' in st.session_state:
                pd.DataFrame(st.session_state['metrics']).T.to_excel(writer, sheet_name='Data Quality Metrics')
            # Outlier Table
            if 'outlier_table' in st.session_state:
                pd.DataFrame(st.session_state['outlier_table']).T.to_excel(writer, sheet_name='Outlier Table')
            # Pattern Table
            if 'pattern_table' in st.session_state:
                pd.DataFrame(st.session_state['pattern_table']).to_excel(writer, sheet_name='Pattern Table')
            # Correlation Matrix
            if 'corr' in st.session_state and st.session_state['corr'] is not None:
                st.session_state['corr'].to_excel(writer, sheet_name='Correlation Matrix')
            # Overall Summary
            summary_data = {
                'Total Rows': [len(df)],
                'Total Columns': [len(df.columns)],
                'Total Nulls': [df.isnull().sum().sum()],
                'Duplicate Rows': [df.duplicated().sum()],
            }
            # Add outlier and data quality notes if available
            notes = []
            if 'outlier_table' in st.session_state:
                notes.append('Outlier columns: ' + ', '.join([col for col, vals in st.session_state['outlier_table'].items() if vals['IQR outlier count'] or vals['Z-score outlier count']]))
            if 'metrics' in st.session_state:
                completeness = pd.DataFrame(st.session_state['metrics']).T['Completeness %'].mean()
                notes.append(f'Average completeness: {completeness:.2f}%')
            summary_df = pd.DataFrame(summary_data)
            if notes:
                summary_df['Notes'] = ['; '.join(notes)]
            summary_df.to_excel(writer, sheet_name='Overall Summary', index=False)
        excel_buffer.seek(0)
        st.download_button('Export Full Report (Excel)', excel_buffer, file_name='full_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.download_button('Download CSV', df.to_csv(index=False).encode('utf-8'), 'report.csv', 'text/csv')

        if st.button('Validate & Show Errors'):
            errors = detect_errors(df, rules)
            st.subheader('Errors Found')
            st.json(errors)
            # Visualize errors
            visualize_data(df, rules, errors)

        if st.button('Apply Corrections'):
            corrected = apply_corrections(df.copy(), rules)
            st.subheader('Corrected Data')
            st.dataframe(corrected.head())
            csv = corrected.to_csv(index=False).encode('utf-8')
            st.download_button('Download Corrected CSV', csv, 'corrected.csv', 'text/csv')

        if st.button('Generate Pytest Code for These Rules'):
            code = generate_pytest_code(rules)
            st.subheader('Generated Pytest Test Code')
            st.code(code, language='python')

if __name__ == '__main__':
    main()
