"""Các bảng kiểm tra chất lượng cho giai đoạn EDA."""
import pandas as pd


def missing_table(df):
    return pd.DataFrame({'missing_count': df.isna().sum(),
                         'missing_percent': (df.isna().mean() * 100).round(2)})\
        .sort_values('missing_percent', ascending=False)


def summary_table(df):
    return df.select_dtypes(include='number').describe().T


def iqr_table(df, columns=('imdb_rating', 'duration_minutes', 'metascore')):
    rows = []
    for c in columns:
        if c not in df:
            continue
        x = pd.to_numeric(df[c], errors='coerce').dropna()
        if x.empty:
            continue
        q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({'column': c, 'Q1': q1, 'Q3': q3, 'IQR': iqr,
                     'lower_bound': lo, 'upper_bound': hi,
                     'outlier_count': int(((x < lo) | (x > hi)).sum()),
                     'valid_count': len(x)})
    return pd.DataFrame(rows)
