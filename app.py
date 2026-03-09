import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from stats_utils import (
    calculate_metric_stats,
    calculate_categorical_stats,
    calculate_mode_for_categorical,
    calculate_crosstab_frequency,
    calculate_crosstab_row_percent,
    calculate_crosstab_col_percent,
)

st.set_page_config(page_title="Statistics Calculator for Students", layout="wide")

st.title("Statistics Calculator for Students")
st.write("Upload a CSV file, define variable scales, and calculate descriptive statistics.")


# -----------------------------
# Session state init
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "scale_df" not in st.session_state:
    st.session_state.scale_df = None

if "applied_scale_df" not in st.session_state:
    st.session_state.applied_scale_df = None


# -----------------------------
# Helper functions
# -----------------------------
def suggest_scale(series: pd.Series, col_name: str) -> str:
    lname = col_name.lower()

    if "id" in lname or "code" in lname or "email" in lname or "name" in lname:
        return "Ignore"

    if pd.api.types.is_numeric_dtype(series):
        unique_vals = pd.to_numeric(series, errors="coerce").dropna().unique()
        unique_vals_sorted = sorted(unique_vals.tolist()) if len(unique_vals) > 0 else []

        # Likert-like suggestion
        if len(unique_vals_sorted) <= 7 and all(v in [1, 2, 3, 4, 5, 6, 7] for v in unique_vals_sorted):
            return "Ordinal"

        return "Metric"

    return "Nominal"


def build_scale_definition(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Variable": df.columns,
        "Scale": [suggest_scale(df[c], c) for c in df.columns]
    })


def get_grouped_variables(scale_df: pd.DataFrame):
    metric_vars = scale_df.loc[scale_df["Scale"] == "Metric", "Variable"].tolist()
    ordinal_vars = scale_df.loc[scale_df["Scale"] == "Ordinal", "Variable"].tolist()
    nominal_vars = scale_df.loc[scale_df["Scale"] == "Nominal", "Variable"].tolist()
    ignore_vars = scale_df.loc[scale_df["Scale"] == "Ignore", "Variable"].tolist()
    return metric_vars, ordinal_vars, nominal_vars, ignore_vars


# -----------------------------
# Upload section
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        MAX_ROWS = 60

        if len(df) > MAX_ROWS:
            st.warning(f"Dataset has {len(df)} rows. Only the first {MAX_ROWS} rows will be used.")
            df = df.head(MAX_ROWS)

        st.session_state.df = df

        current_cols = df.columns.tolist()

        if st.session_state.scale_df is None:
            st.session_state.scale_df = build_scale_definition(df)
        else:
            old_cols = st.session_state.scale_df["Variable"].tolist()
            if old_cols != current_cols:
                st.session_state.scale_df = build_scale_definition(df)
                st.session_state.applied_scale_df = None

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

df = st.session_state.df

if df is not None:
    st.markdown("---")
    st.subheader("Step 1. Define Variable Scales")
    st.caption("Please adjust the scale first, then click Apply Scale Definition.")

    colA, colB = st.columns([1, 5])

    with colA:
        if st.button("Auto Suggest Scales"):
            st.session_state.scale_df = build_scale_definition(df)

    with colB:
        st.info(
            "Metric = continuous numeric values, Ordinal = ordered categories or Likert scales, "
            "Nominal = unordered categories, Ignore = variables not used in analysis."
        )

    edited_scale_df = st.data_editor(
        st.session_state.scale_df,
        hide_index=True,
        width="stretch",
        disabled=["Variable"],
        column_config={
            "Variable": st.column_config.TextColumn("Variable"),
            "Scale": st.column_config.SelectboxColumn(
                "Scale",
                options=["Metric", "Ordinal", "Nominal", "Ignore"],
                required=True,
            ),
        },
        key="scale_editor"
    )

    apply_col1, apply_col2 = st.columns([1, 5])

    with apply_col1:
        if st.button("Apply Scale Definition", type="primary"):
            st.session_state.applied_scale_df = edited_scale_df.copy()
            st.success("Scale definition applied.")

    with apply_col2:
        if st.session_state.applied_scale_df is None:
            st.warning("Please click 'Apply Scale Definition' before continuing.")

    if st.session_state.applied_scale_df is None:
        st.stop()

    applied_scale_df = st.session_state.applied_scale_df.copy()
    metric_vars, ordinal_vars, nominal_vars, ignore_vars = get_grouped_variables(applied_scale_df)

    st.markdown("---")
    st.subheader("Current Classification")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Metric Variables**")
        if metric_vars:
            for v in metric_vars:
                st.checkbox(v, value=True, disabled=True, key=f"metric_display_{v}")
        else:
            st.write("-")

    with c2:
        st.markdown("**Ordinal Variables**")
        if ordinal_vars:
            for v in ordinal_vars:
                st.checkbox(v, value=True, disabled=True, key=f"ordinal_display_{v}")
        else:
            st.write("-")

    with c3:
        st.markdown("**Nominal Variables**")
        if nominal_vars:
            for v in nominal_vars:
                st.checkbox(v, value=True, disabled=True, key=f"nominal_display_{v}")
        else:
            st.write("-")

    with c4:
        st.markdown("**Ignored Variables**")
        if ignore_vars:
            for v in ignore_vars:
                st.checkbox(v, value=True, disabled=True, key=f"ignore_display_{v}")
        else:
            st.write("-")

    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(df, width="stretch")

    st.markdown("---")
    st.subheader("Step 2. Select Variables to Analyze")

    sel1, sel2, sel3 = st.columns(3)

    with sel1:
        selected_metric_vars = st.multiselect(
            "Metric Variables to Analyze",
            metric_vars,
            default=metric_vars,
            key="selected_metric_vars"
        )

    with sel2:
        selected_ordinal_vars = st.multiselect(
            "Ordinal Variables to Analyze",
            ordinal_vars,
            default=ordinal_vars,
            key="selected_ordinal_vars"
        )

    with sel3:
        selected_nominal_vars = st.multiselect(
            "Nominal Variables to Analyze",
            nominal_vars,
            default=nominal_vars,
            key="selected_nominal_vars"
        )

    st.markdown("---")
    st.subheader("Step 3. Choose Statistics")

    metric_options = [
        "Mean",
        "Median",
        "Mode",
        "Sum",
        "Std. Deviation",
        "Variance",
        "Minimum",
        "Maximum",
        "Range",
        "Quartile 1",
        "Quartile 2",
        "Quartile 3",
        "Interquartile Range",
        "Median absolute deviation",
        "Skew",
        "Kurtosis",
        "Number of values",
    ]

    categorical_options = [
        "Frequency",
        "Percentage",
        "Mode",
    ]

    left, right = st.columns(2)

    with left:
        selected_metric_stats = st.multiselect(
            "Metric Statistics",
            metric_options,
            default=["Mean", "Std. Deviation", "Minimum", "Maximum", "Number of values"]
        )

    with right:
        selected_cat_stats = st.multiselect(
            "Ordinal/Nominal Statistics",
            categorical_options,
            default=["Frequency", "Percentage"]
        )

    st.markdown("---")
    st.subheader("Step 4. Calculate")

    calculate_btn = st.button("Calculate Descriptive Statistics", type="primary")

    if calculate_btn:
        if not (selected_metric_vars or selected_ordinal_vars or selected_nominal_vars):
            st.error("Please select at least one variable to analyze.")
            st.stop()

        st.subheader("Descriptive Statistics")

        if selected_metric_vars:
            st.markdown("## Metric Variables")
            for col in selected_metric_vars:
                st.markdown(f"### {col}")

                left_col, right_col = st.columns([1.15, 1])

                with left_col:
                    result_df = calculate_metric_stats(df[col], selected_metric_stats)
                    st.dataframe(result_df, width="content")

                with right_col:
                    numeric_series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(numeric_series) > 0:
                        fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=120)
                        ax.hist(numeric_series, bins=8, edgecolor="black")
                        ax.set_title(col, fontsize=10)
                        ax.set_xlabel("")
                        ax.set_ylabel("Freq", fontsize=9)
                        ax.tick_params(axis="both", labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig, width="content")
                        plt.close(fig)

                st.markdown("---")

        if selected_ordinal_vars:
            st.markdown("## Ordinal Variables")
            for col in selected_ordinal_vars:
                st.markdown(f"### {col}")

                left_col, right_col = st.columns([1.15, 1])

                with left_col:
                    result_df = calculate_categorical_stats(df[col], selected_cat_stats)
                    st.dataframe(result_df, width="content")

                    if "Mode" in selected_cat_stats:
                        st.write(f"**Mode:** {calculate_mode_for_categorical(df[col])}")

                with right_col:
                    chart_data = df[col].astype(str).value_counts()

                    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=120)
                    chart_data.plot(kind="bar", ax=ax)
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel("")
                    ax.set_ylabel("Freq", fontsize=9)
                    ax.tick_params(axis="x", rotation=30, labelsize=8)
                    ax.tick_params(axis="y", labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig, width="content")
                    plt.close(fig)

                st.markdown("---")

        if selected_nominal_vars:
            st.markdown("## Nominal Variables")
            for col in selected_nominal_vars:
                st.markdown(f"### {col}")

                left_col, right_col = st.columns([1.15, 1])

                with left_col:
                    result_df = calculate_categorical_stats(df[col], selected_cat_stats)
                    st.dataframe(result_df, width="content")

                    if "Mode" in selected_cat_stats:
                        st.write(f"**Mode:** {calculate_mode_for_categorical(df[col])}")

                with right_col:
                    chart_data = df[col].astype(str).value_counts()

                    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=120)
                    chart_data.plot(kind="bar", ax=ax)
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel("")
                    ax.set_ylabel("Freq", fontsize=9)
                    ax.tick_params(axis="x", rotation=30, labelsize=8)
                    ax.tick_params(axis="y", labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig, width="content")
                    plt.close(fig)

                st.markdown("---")

        st.success("Calculation completed.")

    st.markdown("---")
    st.subheader("Step 5. Cross-Tabulation")

    crosstab_candidates = selected_ordinal_vars + selected_nominal_vars

    if len(crosstab_candidates) < 2:
        st.info("Please select at least two Ordinal/Nominal variables in Step 2 to use cross-tabulation.")
    else:
        ct1, ct2 = st.columns(2)

        with ct1:
            row_var = st.selectbox(
                "Row Variable",
                crosstab_candidates,
                key="crosstab_row"
            )

        with ct2:
            col_options = [v for v in crosstab_candidates if v != row_var]
            col_var = st.selectbox(
                "Column Variable",
                col_options,
                key="crosstab_col"
            )

        pct1, pct2 = st.columns(2)
        with pct1:
            show_row_pct = st.checkbox("Show Row Percentages", value=True)
        with pct2:
            show_col_pct = st.checkbox("Show Column Percentages", value=False)

        if st.button("Generate Cross-Tabulation", type="primary"):
            st.markdown(f"### Cross-Tabulation: {row_var} × {col_var}")

            freq_df = calculate_crosstab_frequency(df, row_var, col_var)
            st.markdown("**Frequency Table**")
            st.dataframe(freq_df, width="content")

            if show_row_pct:
                row_pct_df = calculate_crosstab_row_percent(df, row_var, col_var)
                st.markdown("**Row Percentage Table (%)**")
                st.dataframe(row_pct_df, width="content")

            if show_col_pct:
                col_pct_df = calculate_crosstab_col_percent(df, row_var, col_var)
                st.markdown("**Column Percentage Table (%)**")
                st.dataframe(col_pct_df, width="content")

            st.markdown("**Stacked Bar Chart**")
            fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=120)
            freq_df.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"{row_var} by {col_var}", fontsize=10)
            ax.set_xlabel(row_var, fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, width="content")
            plt.close(fig)

else:
    st.info("Please upload a CSV file to begin.")