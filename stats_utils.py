import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def calculate_metric_stats(series: pd.Series, selected_stats: list[str]) -> pd.DataFrame:
    s = safe_numeric(series)
    results = {}

    if len(s) == 0:
        return pd.DataFrame({"Statistic": ["No valid numeric data"], "Value": ["-"]})

    if "Number of values" in selected_stats:
        results["Number of values"] = int(s.count())

    if "Mean" in selected_stats:
        results["Mean"] = round(s.mean(), 4)

    if "Median" in selected_stats:
        results["Median"] = round(s.median(), 4)

    if "Mode" in selected_stats:
        mode_vals = s.mode()
        results["Mode"] = ", ".join(map(str, mode_vals.tolist())) if not mode_vals.empty else "-"

    if "Sum" in selected_stats:
        results["Sum"] = round(s.sum(), 4)

    if "Std. Deviation" in selected_stats:
        results["Std. Deviation"] = round(s.std(ddof=1), 4) if len(s) > 1 else 0.0

    if "Variance" in selected_stats:
        results["Variance"] = round(s.var(ddof=1), 4) if len(s) > 1 else 0.0

    if "Minimum" in selected_stats:
        results["Minimum"] = round(s.min(), 4)

    if "Maximum" in selected_stats:
        results["Maximum"] = round(s.max(), 4)

    if "Range" in selected_stats:
        results["Range"] = round(s.max() - s.min(), 4)

    if "Quartile 1" in selected_stats:
        results["Quartile 1"] = round(s.quantile(0.25), 4)

    if "Quartile 2" in selected_stats:
        results["Quartile 2"] = round(s.quantile(0.50), 4)

    if "Quartile 3" in selected_stats:
        results["Quartile 3"] = round(s.quantile(0.75), 4)

    if "Interquartile Range" in selected_stats:
        results["Interquartile Range"] = round(s.quantile(0.75) - s.quantile(0.25), 4)

    if "Median absolute deviation" in selected_stats:
        med = s.median()
        results["Median absolute deviation"] = round(np.median(np.abs(s - med)), 4)

    if "Skew" in selected_stats:
        results["Skew"] = round(skew(s, bias=False), 4) if len(s) > 2 else "-"

    if "Kurtosis" in selected_stats:
        results["Kurtosis"] = round(kurtosis(s, bias=False), 4) if len(s) > 3 else "-"

    return pd.DataFrame({
        "Statistic": list(results.keys()),
        "Value": list(results.values())
    })


def calculate_categorical_stats(series: pd.Series, selected_stats: list[str]) -> pd.DataFrame:
    s = series.copy()
    s = s.fillna("Missing").astype(str)

    freq = s.value_counts(dropna=False)
    perc = (freq / freq.sum() * 100).round(2)

    result = pd.DataFrame({
        "Category": freq.index,
        "Frequency": freq.values
    })

    if "Percentage" in selected_stats:
        result["Percentage"] = perc.values

    return result


def calculate_mode_for_categorical(series: pd.Series):
    s = series.copy()
    s = s.fillna("Missing").astype(str)

    m = s.mode()
    if m.empty:
        return "-"
    return ", ".join(m.tolist())


def calculate_crosstab_frequency(df: pd.DataFrame, row_var: str, col_var: str) -> pd.DataFrame:
    row_series = df[row_var].fillna("Missing")
    col_series = df[col_var].fillna("Missing")
    return pd.crosstab(row_series, col_series, dropna=False)


def calculate_crosstab_row_percent(df: pd.DataFrame, row_var: str, col_var: str) -> pd.DataFrame:
    freq = calculate_crosstab_frequency(df, row_var, col_var)
    row_totals = freq.sum(axis=1)
    row_pct = freq.div(row_totals, axis=0) * 100
    return row_pct.round(2)


def calculate_crosstab_col_percent(df: pd.DataFrame, row_var: str, col_var: str) -> pd.DataFrame:
    freq = calculate_crosstab_frequency(df, row_var, col_var)
    col_totals = freq.sum(axis=0)
    col_pct = freq.div(col_totals, axis=1) * 100
    return col_pct.round(2)