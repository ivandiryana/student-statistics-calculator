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

from openai import OpenAI


# -----------------------------
# OpenAI
# -----------------------------
client = None
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Statistics Calculator for Students", layout="wide")


# -----------------------------
# Session state init
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "scale_df" not in st.session_state:
    st.session_state.scale_df = None

if "applied_scale_df" not in st.session_state:
    st.session_state.applied_scale_df = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "calculator"

if "calc_done" not in st.session_state:
    st.session_state.calc_done = False

if "metric_results_store" not in st.session_state:
    st.session_state.metric_results_store = {}

if "ordinal_results_store" not in st.session_state:
    st.session_state.ordinal_results_store = {}

if "nominal_results_store" not in st.session_state:
    st.session_state.nominal_results_store = {}

if "selected_metric_vars_store" not in st.session_state:
    st.session_state.selected_metric_vars_store = []

if "selected_ordinal_vars_store" not in st.session_state:
    st.session_state.selected_ordinal_vars_store = []

if "selected_nominal_vars_store" not in st.session_state:
    st.session_state.selected_nominal_vars_store = []

if "selected_metric_stats_store" not in st.session_state:
    st.session_state.selected_metric_stats_store = []

if "selected_cat_stats_store" not in st.session_state:
    st.session_state.selected_cat_stats_store = []

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []


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


def serialize_df_for_prompt(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "Tidak ada data."
    preview_df = df.head(max_rows).copy()
    return preview_df.to_csv(index=False)


def serialize_full_df_for_prompt(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Tidak ada data."
    return df.to_csv(index=False)


def build_full_results_for_welcome() -> str:
    lines = []

    if st.session_state.metric_results_store:
        lines.append("## Hasil Statistik Deskriptif - Metric")
        for col, result_df in st.session_state.metric_results_store.items():
            lines.append(f"\n### {col}")
            try:
                result_dict = result_df.iloc[:, 0].to_dict()
                for stat_name, value in result_dict.items():
                    lines.append(f"- {stat_name}: {value}")
            except Exception:
                lines.append(str(result_df))

    if st.session_state.ordinal_results_store:
        lines.append("\n## Hasil Statistik Deskriptif - Ordinal")
        for col, result_df in st.session_state.ordinal_results_store.items():
            lines.append(f"\n### {col}")
            try:
                for _, row in result_df.iterrows():
                    row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                    lines.append(f"- {row_text}")
            except Exception:
                lines.append(str(result_df))

            if st.session_state.df is not None and col in st.session_state.df.columns:
                try:
                    mode_val = calculate_mode_for_categorical(st.session_state.df[col])
                    lines.append(f"- Mode: {mode_val}")
                except Exception:
                    pass

    if st.session_state.nominal_results_store:
        lines.append("\n## Hasil Statistik Deskriptif - Nominal")
        for col, result_df in st.session_state.nominal_results_store.items():
            lines.append(f"\n### {col}")
            try:
                for _, row in result_df.iterrows():
                    row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                    lines.append(f"- {row_text}")
            except Exception:
                lines.append(str(result_df))

            if st.session_state.df is not None and col in st.session_state.df.columns:
                try:
                    mode_val = calculate_mode_for_categorical(st.session_state.df[col])
                    lines.append(f"- Mode: {mode_val}")
                except Exception:
                    pass

    if not lines:
        return "Belum ada hasil statistik yang dihitung."

    return "\n".join(lines)


def build_ai_context() -> str:
    df = st.session_state.df
    applied_scale_df = st.session_state.applied_scale_df

    lines = []

    lines.append("KONTEKS DATASET")
    lines.append(f"Jumlah baris: {len(df) if df is not None else 0}")
    lines.append(f"Jumlah kolom: {len(df.columns) if df is not None else 0}")
    lines.append("")

    if applied_scale_df is not None:
        lines.append("SKALA VARIABEL")
        for _, row in applied_scale_df.iterrows():
            lines.append(f"- {row['Variable']}: {row['Scale']}")
        lines.append("")

    lines.append("DATA MENTAH LENGKAP")
    lines.append(serialize_full_df_for_prompt(df))
    lines.append("")

    if st.session_state.metric_results_store:
        lines.append("HASIL STATISTIK METRIC")
        for col, result_df in st.session_state.metric_results_store.items():
            lines.append(f"[{col}]")
            try:
                result_dict = result_df.iloc[:, 0].to_dict()
                for k, v in result_dict.items():
                    lines.append(f"- {k}: {v}")
            except Exception:
                lines.append(str(result_df))
            lines.append("")

    if st.session_state.ordinal_results_store:
        lines.append("HASIL STATISTIK ORDINAL")
        for col, result_df in st.session_state.ordinal_results_store.items():
            lines.append(f"[{col}]")
            try:
                lines.append(result_df.to_csv(index=False))
            except Exception:
                lines.append(str(result_df))
            mode_val = calculate_mode_for_categorical(df[col]) if df is not None and col in df.columns else "-"
            lines.append(f"Mode: {mode_val}")
            lines.append("")

    if st.session_state.nominal_results_store:
        lines.append("HASIL STATISTIK NOMINAL")
        for col, result_df in st.session_state.nominal_results_store.items():
            lines.append(f"[{col}]")
            try:
                lines.append(result_df.to_csv(index=False))
            except Exception:
                lines.append(str(result_df))
            mode_val = calculate_mode_for_categorical(df[col]) if df is not None and col in df.columns else "-"
            lines.append(f"Mode: {mode_val}")
            lines.append("")

    return "\n".join(lines)


def get_column_values_for_ai(column_name: str) -> str:
    df = st.session_state.df

    if df is None or column_name not in df.columns:
        return "Kolom tidak ditemukan."

    series = df[column_name].dropna()

    if len(series) == 0:
        return "Kolom kosong."

    lines = [f"DETAIL NILAI KOLOM: {column_name}"]

    value_counts = series.astype(str).value_counts()
    lines.append("Distribusi nilai:")
    for val, freq in value_counts.items():
        lines.append(f"- {val}: {freq}")

    lines.append("")
    lines.append("Nilai mentah:")
    lines.append(", ".join(series.astype(str).tolist()))

    return "\n".join(lines)


def detect_relevant_column(user_question: str):
    df = st.session_state.df
    if df is None:
        return None

    q = user_question.lower()

    for col in df.columns:
        if col.lower() in q:
            return col

    all_selected = (
        st.session_state.selected_metric_vars_store
        + st.session_state.selected_ordinal_vars_store
        + st.session_state.selected_nominal_vars_store
    )

    if len(all_selected) == 1:
        return all_selected[0]

    return None


def generate_ai_welcome_message() -> str:
    full_results_text = build_full_results_for_welcome()

    intro_text = (
        "## Hasil Statistik Deskriptif\n\n"
        f"{full_results_text}\n\n"
    )

    ai_intro = (
        "Saya Professor Ivan, AI Asisten Pak Ivan. "
        "Saya sudah menerima dataset mentah dan seluruh hasil statistik deskriptif Anda.\n\n"
        "Silakan tanyakan apa saja tentang hasil ini.\n\n"
        "Contoh pertanyaan:\n\n"
        "• Bagaimana mean OverallSatisfaction dihitung dari data ini?\n"
        "• Apa arti standar deviasi DeliverySpeed yang kecil?\n"
        "• Mengapa nilai maksimum WebsiteEase adalah 5?\n"
        "• Berapa jumlah laki-laki yang sering berbelanja?\n"
        "• Bagaimana menjelaskan hasil ini di laporan?"
    )

    return intro_text + ai_intro

def is_descriptive_stats_question(user_question: str) -> bool:
    q = user_question.lower()

    allowed_keywords = [
        "mean", "rata-rata", "average",
        "median",
        "mode", "modus",
        "sum", "jumlah",
        "std", "standar deviasi", "standard deviation",
        "variance", "varians",
        "minimum", "maximum", "maksimum", "minimum",
        "range", "rentang",
        "quartile", "kuartil", "q1", "q2", "q3",
        "interquartile", "iqr",
        "mad", "median absolute deviation",
        "skew", "skewness",
        "kurtosis",
        "frequency", "frekuensi",
        "percentage", "persentase", "proporsi",
        "cross tab", "crosstab", "cross-tab", "tabulasi silang",
        "berapa jumlah", "berapa banyak", "berapa persen",
        "berapa nilai",
        "bagaimana menghitung", "cara menghitung", "rumus",
        "apa arti", "interpretasi", "jelaskan hasil",
        "variabel", "kolom", "data ini", "dataset ini",
        "laki", "perempuan", "male", "female",
    ]

    blocked_keywords = [
        "regresi", "regression",
        "anova",
        "korelasi", "correlation",
        "uji hipotesis", "hypothesis",
        "p value", "p-value",
        "signifikan", "significance",
        "causal", "causality", "sebab akibat",
        "prediksi", "prediction", "forecast",
        "machine learning",
        "clustering",
        "rekomendasi bisnis", "strategi bisnis", "marketing strategy",
        "kesimpulan umum di luar data",
        "diagnosis",
        "medis", "penyakit",
        "hukum", "legal",
        "keuangan pribadi", "investasi",
        "politik",
    ]

    if any(bk in q for bk in blocked_keywords):
        return False

    return any(ak in q for ak in allowed_keywords)

def descriptive_scope_guard_message() -> str:
    return (
        "Maaf, saya hanya bisa membantu menjawab pertanyaan yang masih berada dalam ruang lingkup "
        "statistik deskriptif dan harus berdasarkan dataset yang sedang dianalisis.\n\n"
        "Saya bisa membantu untuk hal-hal seperti:\n"
        "- cara menghitung mean, median, mode, standar deviasi, varians, range, kuartil, skewness, atau kurtosis\n"
        "- arti hasil statistik deskriptif\n"
        "- frekuensi, persentase, dan tabulasi silang sederhana\n"
        "- jumlah atau persentase responden dengan kondisi tertentu berdasarkan data yang ada\n\n"
        "Silakan ajukan pertanyaan yang terkait langsung dengan hasil statistik deskriptif atau data ini."
    )

def ask_ai_about_data(user_question: str) -> str:
    if client is None:
        return "AI belum tersedia. Pastikan OPENAI_API_KEY sudah diset di Streamlit secrets."

    if not is_descriptive_stats_question(user_question):
        return descriptive_scope_guard_message()

    context_text = build_ai_context()

    relevant_col = detect_relevant_column(user_question)
    column_detail = ""

    if relevant_col is not None:
        column_detail = get_column_values_for_ai(relevant_col)
    else:
        column_detail = (
            "Tidak ada kolom spesifik yang berhasil dideteksi dari pertanyaan.\n"
            "Jika ingin langkah hitung dari data nyata, pengguna sebaiknya menyebutkan nama variabel."
        )

    system_prompt = """
Anda adalah tutor statistik untuk mahasiswa.

BATASAN WAJIB:
- Anda hanya boleh menjawab hal-hal yang terkait statistik deskriptif
- Anda hanya boleh menjawab berdasarkan dataset, data mentah, dan hasil statistik yang diberikan
- Anda tidak boleh menjawab topik di luar statistik deskriptif
- Anda tidak boleh menjawab inferensi statistik, uji hipotesis, regresi, korelasi, prediksi, machine learning, strategi bisnis umum, atau topik lain di luar konteks data ini
- jika pertanyaan di luar ruang lingkup tersebut, jawab bahwa Anda hanya bisa membantu statistik deskriptif berbasis data yang tersedia

Aturan jawaban:
- selalu utamakan konteks dataset, data mentah, dan hasil statistik yang diberikan
- gunakan data mentah jika pertanyaan menyangkut hubungan antar variabel atau jumlah responden dengan kondisi tertentu
- gunakan hasil statistik deskriptif jika pertanyaan menyangkut mean, median, standar deviasi, variance, skewness, kurtosis, mode, frequency, atau percentage
- jangan langsung memberi penjelasan teori umum yang panjang
- jika pengguna bertanya tentang mean, median, standar deviasi, variance, skewness, kurtosis, mode, frequency, atau percentage, jelaskan dengan mengacu pada variabel dan hasil yang ada
- jika menjelaskan rumus, hubungkan rumus dengan kolom yang relevan
- jika data frekuensi tersedia, gunakan data frekuensi itu untuk menjelaskan perhitungan
- jika nilai mentah tersedia, gunakan nilai mentah tersebut
- tampilkan rumus dan langkah hitung menggunakan data nyata jika memungkinkan
- jika pertanyaan meminta jumlah atau kondisi gabungan, hitung berdasarkan data mentah yang tersedia
- jangan mengatakan data tidak tersedia jika sebenarnya data mentah tersedia dalam konteks
- jangan mengarang kategori, skala, atau nilai yang tidak ada dalam konteks
- jika informasi belum cukup untuk memberi contoh angka rinci, katakan itu secara jujur
- gunakan bahasa Indonesia yang sederhana, jelas, dan bersifat mengajar
- jawaban harus terasa personal terhadap data mahasiswa, bukan seperti definisi buku teks
- fokus pada pembelajaran, bukan hanya memberi jawaban singkat
- jangan gunakan format LaTeX atau notasi matematika seperti \\[ \\], \\( \\), \\frac, \\sum
- jika menuliskan rumus, gunakan format teks biasa
- contoh format yang benar:
  Mean = ((2 x 1) + (3 x 7) + (4 x 25) + (5 x 17)) / 50 = 4.16
- jika memungkinkan, susun jawaban dengan urutan:
  1. Cara hitung
  2. Langkah perhitungan
  3. Hasil
  4. Artinya
"""

    user_prompt = f"""
Berikut adalah konteks data dan hasil statistik mahasiswa:

{context_text}

Detail kolom yang kemungkinan relevan:
{column_detail}

Jawablah pertanyaan berikut dengan mengacu pada konteks di atas.
Anda hanya boleh menjawab dalam ruang lingkup statistik deskriptif dan hanya berdasarkan data yang tersedia.
Jika pertanyaannya tentang perhitungan, tampilkan langkah hitung dari data yang tersedia.
Tulis semua rumus dalam format teks biasa, bukan LaTeX.

Pertanyaan mahasiswa:
{user_question}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI tidak dapat memberikan jawaban saat ini. Detail error: {e}"
# =====================================================
# PAGE: AI CHAT
# =====================================================
if st.session_state.current_page == "ai_chat":
    st.title("AI Statistics Tutor")
    st.write("Tanyakan apa saja tentang dataset dan hasil statistik Anda.")

    top_left, top_right = st.columns([1, 4])

    with top_left:
        if st.button("← Kembali ke Kalkulator"):
            st.session_state.current_page = "calculator"
            st.rerun()

    with top_right:
        st.info(
            "Contoh pertanyaan: 'Bagaimana mean OverallSatisfaction dihitung dari data ini?' "
            "atau 'Berapa jumlah laki-laki yang sering berbelanja?'"
        )

    if not st.session_state.ai_chat_history:
        welcome_msg = generate_ai_welcome_message()
        st.session_state.ai_chat_history = [
            {"role": "assistant", "content": welcome_msg}
        ]

    with st.expander("Lihat konteks yang dikirim ke AI"):
        st.text(build_ai_context())

    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Tulis pertanyaan Anda tentang data atau hasil statistik...")

    if user_input:
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI sedang menjawab..."):
                ai_response = ask_ai_about_data(user_input)
                st.markdown(ai_response)

        st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})

    st.stop()


# =====================================================
# PAGE: CALCULATOR
# =====================================================
st.title("Statistics Calculator for Students")
st.write("Upload a CSV file, define variable scales, and calculate descriptive statistics.")


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
                st.session_state.calc_done = False
                st.session_state.metric_results_store = {}
                st.session_state.ordinal_results_store = {}
                st.session_state.nominal_results_store = {}
                st.session_state.ai_chat_history = []

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
            st.session_state.calc_done = False
            st.session_state.metric_results_store = {}
            st.session_state.ordinal_results_store = {}
            st.session_state.nominal_results_store = {}
            st.session_state.ai_chat_history = []
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

        st.session_state.metric_results_store = {}
        st.session_state.ordinal_results_store = {}
        st.session_state.nominal_results_store = {}
        st.session_state.selected_metric_vars_store = selected_metric_vars
        st.session_state.selected_ordinal_vars_store = selected_ordinal_vars
        st.session_state.selected_nominal_vars_store = selected_nominal_vars
        st.session_state.selected_metric_stats_store = selected_metric_stats
        st.session_state.selected_cat_stats_store = selected_cat_stats

        if selected_metric_vars:
            for col in selected_metric_vars:
                st.session_state.metric_results_store[col] = calculate_metric_stats(df[col], selected_metric_stats)

        if selected_ordinal_vars:
            for col in selected_ordinal_vars:
                st.session_state.ordinal_results_store[col] = calculate_categorical_stats(df[col], selected_cat_stats)

        if selected_nominal_vars:
            for col in selected_nominal_vars:
                st.session_state.nominal_results_store[col] = calculate_categorical_stats(df[col], selected_cat_stats)

        st.session_state.calc_done = True
        st.session_state.ai_chat_history = []

    if st.session_state.calc_done:
        st.subheader("Descriptive Statistics")

        if st.session_state.selected_metric_vars_store:
            st.markdown("## Metric Variables")
            for col in st.session_state.selected_metric_vars_store:
                st.markdown(f"### {col}")

                left_col, right_col = st.columns([1.15, 1])

                with left_col:
                    result_df = st.session_state.metric_results_store[col]
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

        if st.session_state.selected_ordinal_vars_store:
            st.markdown("## Ordinal Variables")
            for col in st.session_state.selected_ordinal_vars_store:
                st.markdown(f"### {col}")

                left_col, right_col = st.columns([1.15, 1])

                with left_col:
                    result_df = st.session_state.ordinal_results_store[col]
                    st.dataframe(result_df, width="content")

                    if "Mode" in st.session_state.selected_cat_stats_store:
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

        if st.session_state.selected_nominal_vars_store:
            st.markdown("## Nominal Variables")
            for col in st.session_state.selected_nominal_vars_store:
                st.markdown(f"### {col}")

                left_col, right_col = st.columns([1.15, 1])

                with left_col:
                    result_df = st.session_state.nominal_results_store[col]
                    st.dataframe(result_df, width="content")

                    if "Mode" in st.session_state.selected_cat_stats_store:
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

        ai_col1, ai_col2 = st.columns([1, 5])
        with ai_col1:
            if st.button("Diskusi dengan AI", type="primary"):
                st.session_state.current_page = "ai_chat"

                if not st.session_state.ai_chat_history:
                    welcome_msg = generate_ai_welcome_message()
                    st.session_state.ai_chat_history = [
                        {"role": "assistant", "content": welcome_msg}
                    ]

                st.rerun()

        with ai_col2:
            st.caption(
                "Buka AI Tutor untuk bertanya tentang cara menghitung mean, median, standar deviasi, "
                "hubungan antar variabel, dan jika memungkinkan AI akan menunjukkan langkah hitung dari data nyata."
            )

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