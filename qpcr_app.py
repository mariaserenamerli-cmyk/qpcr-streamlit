import io
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def find_col(df: pd.DataFrame, candidates):
    cols = {norm(c): c for c in df.columns}
    for cand in candidates:
        k = norm(cand)
        if k in cols:
            return cols[k]
    return None

def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def sort_groups(groups):
    def key(g):
        g = str(g)
        m = re.match(r"^\s*(\d+)", g)
        if m:
            return (0, int(m.group(1)), g)
        if g.upper().startswith("H2O"):
            return (1, 10**9, g)
        return (2, 10**9, g)
    return sorted(list(groups), key=key)

def barplot_mean_sd(df_summary: pd.DataFrame, label_col: str, mean_col: str, sd_col: str, title: str):
    labels = df_summary[label_col].astype(str).tolist()
    means = df_summary[mean_col].to_numpy(dtype=float)
    sds = df_summary[sd_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))  # stile "Excel" meno schiacciato
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(mean_col)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig

def fig_to_jpeg_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def norm_well(x) -> str:
    """
    Normalizza Well ID per evitare mismatch tipo 12 vs 12.0 vs "12 ".
    Lascia invariati i Well alfanumerici tipo A1.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    # se è un numero (o stringa numerica), convertilo a intero se possibile
    try:
        fx = float(s)
        if fx.is_integer():
            return str(int(fx))
        return str(fx)
    except Exception:
        return s


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="qPCR Summary + Layout", layout="wide")
st.title("qPCR — Layout + Summary (mean ± SD)")

st.markdown(
    """
Carica i dati grezzi (CSV o Excel), imposta lo **schema Well → Loaded** e ottieni tabella + grafici.

**Regola positività:** NEGATIVO solo se **Amplitude < soglia** **e** **Slope < soglia**.  
(Quindi è POSITIVO se Amplitude ≥ soglia **oppure** Slope ≥ soglia.)
"""
)

with st.sidebar:
    st.header("Parametri")
    ampl_thr = st.number_input("Soglia Amplitude (negativo se <)", value=10.0, step=0.5)
    slope_thr = st.number_input("Soglia Slope (negativo se <)", value=1.0, step=0.1)
    invalid_cq_text = st.text_input("Valori Cq da ignorare (separati da virgola)", value="-1,2,0")
    show_negative_plots = st.checkbox("Mostra anche grafici negativi (Amplitude/Slope)", value=False)
    include_unmapped = st.checkbox("Includi UNMAPPED nel riepilogo", value=False)

uploaded = st.file_uploader("Carica RAW file (.csv / .xlsx)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.stop()

# Load data
try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Errore leggendo il file: {e}")
    st.stop()

st.subheader("Preview dati (prime righe)")
st.dataframe(df_raw.head(20), use_container_width=True)

# Auto-detect columns
col_well_guess = find_col(df_raw, ["Well ID", "Well", "WellID", "Well Id"])
col_loaded_guess = find_col(df_raw, ["Loaded", "Group"])
col_assay_guess = find_col(df_raw, ["Assay"])
col_channel_guess = find_col(df_raw, ["Channel"])
col_cq_guess = find_col(df_raw, ["Cq", "Ct"])
col_ampl_guess = find_col(df_raw, ["Ampl.", "Amplitude", "Ampl"])
col_slope_guess = find_col(df_raw, ["Slope"])

# Column mapping UI
st.subheader("Mappatura colonne (controlla che sia giusto)")
cols = list(df_raw.columns)

c1, c2, c3 = st.columns(3)
with c1:
    col_well = st.selectbox("Colonna Well", options=cols, index=cols.index(col_well_guess) if col_well_guess in cols else 0)
    col_cq = st.selectbox("Colonna Cq", options=cols, index=cols.index(col_cq_guess) if col_cq_guess in cols else 0)
with c2:
    col_ampl = st.selectbox("Colonna Amplitude", options=cols, index=cols.index(col_ampl_guess) if col_ampl_guess in cols else 0)
    col_slope = st.selectbox("Colonna Slope", options=cols, index=cols.index(col_slope_guess) if col_slope_guess in cols else 0)
with c3:
    col_assay = st.selectbox("Colonna Assay (opzionale)", options=["(nessuna)"] + cols,
                             index=(["(nessuna)"] + cols).index(col_assay_guess) if col_assay_guess in cols else 0)
    col_channel = st.selectbox("Colonna Channel (opzionale)", options=["(nessuna)"] + cols,
                               index=(["(nessuna)"] + cols).index(col_channel_guess) if col_channel_guess in cols else 0)
    col_loaded = st.selectbox("Colonna Loaded/Group nel file (opzionale)", options=["(nessuna)"] + cols,
                              index=(["(nessuna)"] + cols).index(col_loaded_guess) if col_loaded_guess in cols else 0)

# Working df
df = df_raw.copy()

# Normalize Well in RAW + DF (IMPORTANT)
df["_well_norm"] = df[col_well].map(norm_well)
df_raw["_well_norm"] = df_raw[col_well].map(norm_well)

# Convert numeric columns
df[col_cq] = to_numeric(df[col_cq])
df[col_ampl] = to_numeric(df[col_ampl])
df[col_slope] = to_numeric(df[col_slope])

# Optional filtering by Assay/Channel (done AFTER mapping is prepared)
if col_assay != "(nessuna)":
    assays = sorted(df[col_assay].dropna().astype(str).unique().tolist())
    assay_pick = st.selectbox("Seleziona Assay", options=["ALL"] + assays, index=0) if assays else "ALL"
    if assay_pick != "ALL":
        df = df[df[col_assay].astype(str) == assay_pick]
else:
    assay_pick = "ALL"

if col_channel != "(nessuna)":
    channels = sorted(df[col_channel].dropna().astype(str).unique().tolist())
    ch_pick = st.selectbox("Seleziona Channel", options=["ALL"] + channels, index=0) if channels else "ALL"
    if ch_pick != "ALL":
        df = df[df[col_channel].astype(str) == ch_pick]
else:
    ch_pick = "ALL"

st.info(f"Righe dopo filtri: {len(df)} | Assay: {assay_pick} | Channel: {ch_pick}")

# -------------------------
# Layout editor (Well → Loaded)  (build from ALL wells in original file)
# -------------------------
st.subheader("Schema di caricamento (Layout) — edita Well → Loaded")

all_wells = df_raw["_well_norm"].dropna()
all_wells = [w for w in all_wells.unique().tolist() if str(w).strip() != ""]
all_wells = sorted(all_wells, key=lambda x: (len(str(x)), str(x)))  # ordine semplice

layout_df = pd.DataFrame({"Well": all_wells})

# Pre-fill Loaded if present in file, otherwise empty
if col_loaded != "(nessuna)":
    tmp = df_raw[["_well_norm", col_loaded]].copy()
    tmp[col_loaded] = tmp[col_loaded].astype(str).str.strip()
    tmp = tmp.drop_duplicates("_well_norm")
    preload = dict(zip(tmp["_well_norm"], tmp[col_loaded]))
    layout_df["Loaded"] = layout_df["Well"].map(lambda w: preload.get(w, ""))
else:
    layout_df["Loaded"] = ""

st.caption("Tip: se due (o più) wells hanno lo stesso **Loaded** (es. 100_AC), verranno mediati insieme automaticamente.")

# Persist editor in session
edited_layout = st.data_editor(
    layout_df,
    use_container_width=True,
    num_rows="fixed",
    key="layout_editor"
)

# Build mapping dict (skip empty Loaded)
edited_layout["Well"] = edited_layout["Well"].map(norm_well)
edited_layout["Loaded"] = edited_layout["Loaded"].astype(str).str.strip()

map_dict = {
    w: l for (w, l) in zip(edited_layout["Well"], edited_layout["Loaded"])
    if str(w).strip() != "" and str(l).strip() != ""
}

# Apply mapping to filtered df
df["Loaded"] = df["_well_norm"].map(map_dict).fillna("UNMAPPED")

unmapped_rows = int((df["Loaded"] == "UNMAPPED").sum())
if unmapped_rows > 0:
    st.warning(f"⚠️ Attenzione: {unmapped_rows} righe sono ancora UNMAPPED (schema incompleto o mismatch).")
    unmapped_wells = df.loc[df["Loaded"] == "UNMAPPED", "_well_norm"].value_counts().head(20)
    st.write("Esempi di Well UNMAPPED (top 20):")
    st.dataframe(unmapped_wells.rename("count"), use_container_width=True)
else:
    st.success("✅ Schema applicato: nessun UNMAPPED nel dataset filtrato.")

# Optionally exclude UNMAPPED from calculations
df_calc = df.copy()
if not include_unmapped:
    df_calc = df_calc[df_calc["Loaded"] != "UNMAPPED"].copy()

# -------------------------
# Positivity rule
# -------------------------
ampl = df_calc[col_ampl].fillna(-np.inf)
slope = df_calc[col_slope].fillna(-np.inf)

pos_mask = (ampl >= ampl_thr) | (slope >= slope_thr)
neg_mask = ~pos_mask

# Cq valid for positives only (ignore invalid placeholders)
invalid_cq_vals = []
for v in invalid_cq_text.split(","):
    v = v.strip()
    if not v:
        continue
    try:
        invalid_cq_vals.append(float(v))
    except ValueError:
        pass

cq_valid = pos_mask & df_calc[col_cq].notna()
if invalid_cq_vals:
    cq_valid = cq_valid & ~df_calc[col_cq].isin(invalid_cq_vals)
cq_valid = cq_valid & (df_calc[col_cq] > 0)

# -------------------------
# Summary by Loaded
# -------------------------
summaries = []
for g, sub in df_calc.groupby("Loaded", dropna=False):
    total = len(sub)
    if total == 0:
        continue

    n_pos = int(pos_mask.loc[sub.index].sum())
    det = (n_pos / total * 100.0) if total else 0.0

    # Cq only from positives
    cq_vals = sub.loc[cq_valid.loc[sub.index], col_cq]
    cq_mean = cq_vals.mean()
    cq_sd = cq_vals.std(ddof=1)

    # all
    ampl_all = sub[col_ampl]
    slope_all = sub[col_slope]
    ampl_mean_all = ampl_all.mean()
    ampl_sd_all = ampl_all.std(ddof=1)
    slope_mean_all = slope_all.mean()
    slope_sd_all = slope_all.std(ddof=1)

    # negatives only
    ampl_neg = sub.loc[neg_mask.loc[sub.index], col_ampl]
    slope_neg = sub.loc[neg_mask.loc[sub.index], col_slope]
    ampl_mean_neg = ampl_neg.mean()
    ampl_sd_neg = ampl_neg.std(ddof=1)
    slope_mean_neg = slope_neg.mean()
    slope_sd_neg = slope_neg.std(ddof=1)

    summaries.append({
        "Loaded": g,
        "Cq mean (pos only)": cq_mean,
        "Cq SD (pos only)": cq_sd,
        "Ampl mean (all)": ampl_mean_all,
        "Ampl SD (all)": ampl_sd_all,
        "Slope mean (all)": slope_mean_all,
        "Slope SD (all)": slope_sd_all,
        "Ampl mean (neg)": ampl_mean_neg,
        "Ampl SD (neg)": ampl_sd_neg,
        "Slope mean (neg)": slope_mean_neg,
        "Slope SD (neg)": slope_sd_neg,
        "Pos/Total": f"{n_pos}/{total}",
        "Detection %": det,
        "N": total
    })

summary_df = pd.DataFrame(summaries)

if len(summary_df) == 0:
    st.error("Nessun gruppo calcolabile. Probabile che tutto sia UNMAPPED oppure Loaded vuoto.")
    st.stop()

summary_df["Loaded"] = pd.Categorical(
    summary_df["Loaded"],
    categories=sort_groups(summary_df["Loaded"].unique()),
    ordered=True
)
summary_df = summary_df.sort_values("Loaded").reset_index(drop=True)

st.subheader("Tabella riepilogo")
st.dataframe(summary_df, use_container_width=True)

# Download summary CSV
csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Scarica tabella CSV",
    data=csv_bytes,
    file_name=f"summary_{assay_pick}_{ch_pick}.csv".replace(" ", "_"),
    mime="text/csv"
)

# -------------------------
# Plots (bar mean ± SD)
# -------------------------
st.subheader("Istogrammi (media ± SD)")

plot_df = summary_df.copy()
for c in ["Cq SD (pos only)", "Ampl SD (all)", "Slope SD (all)", "Ampl SD (neg)", "Slope SD (neg)"]:
    if c in plot_df.columns:
        plot_df[c] = plot_df[c].fillna(0)

# Cq plot (positives)
fig_cq = barplot_mean_sd(
    plot_df, "Loaded",
    "Cq mean (pos only)", "Cq SD (pos only)",
    title=f"Cq (positivi) — mean ± SD | Assay={assay_pick} | Channel={ch_pick}"
)
st.pyplot(fig_cq, use_container_width=True)
st.download_button(
    "Scarica JPEG Cq",
    data=fig_to_jpeg_bytes(fig_cq),
    file_name=f"bar_Cq_{assay_pick}_{ch_pick}.jpeg".replace(" ", "_"),
    mime="image/jpeg"
)

# Amplitude plot (all)
fig_ampl = barplot_mean_sd(
    plot_df, "Loaded",
    "Ampl mean (all)", "Ampl SD (all)",
    title=f"Amplitude (tutti) — mean ± SD | Assay={assay_pick} | Channel={ch_pick}"
)
st.pyplot(fig_ampl, use_container_width=True)
st.download_button(
    "Scarica JPEG Amplitude",
    data=fig_to_jpeg_bytes(fig_ampl),
    file_name=f"bar_Amplitude_{assay_pick}_{ch_pick}.jpeg".replace(" ", "_"),
    mime="image/jpeg"
)

# Slope plot (all)
fig_slope = barplot_mean_sd(
    plot_df, "Loaded",
    "Slope mean (all)", "Slope SD (all)",
    title=f"Slope (tutti) — mean ± SD | Assay={assay_pick} | Channel={ch_pick}"
)
st.pyplot(fig_slope, use_container_width=True)
st.download_button(
    "Scarica JPEG Slope",
    data=fig_to_jpeg_bytes(fig_slope),
    file_name=f"bar_Slope_{assay_pick}_{ch_pick}.jpeg".replace(" ", "_"),
    mime="image/jpeg"
)

# Optional negative-only plots
if show_negative_plots:
    st.subheader("Grafici (solo negativi)")

    fig_ampl_neg = barplot_mean_sd(
        plot_df, "Loaded",
        "Ampl mean (neg)", "Ampl SD (neg)",
        title=f"Amplitude (negativi) — mean ± SD | Assay={assay_pick} | Channel={ch_pick}"
    )
    st.pyplot(fig_ampl_neg, use_container_width=True)
    st.download_button(
        "Scarica JPEG Amplitude NEG",
        data=fig_to_jpeg_bytes(fig_ampl_neg),
        file_name=f"bar_Amplitude_NEG_{assay_pick}_{ch_pick}.jpeg".replace(" ", "_"),
        mime="image/jpeg"
    )

    fig_slope_neg = barplot_mean_sd(
        plot_df, "Loaded",
        "Slope mean (neg)", "Slope SD (neg)",
        title=f"Slope (negativi) — mean ± SD | Assay={assay_pick} | Channel={ch_pick}"
    )
    st.pyplot(fig_slope_neg, use_container_width=True)
    st.download_button(
        "Scarica JPEG Slope NEG",
        data=fig_to_jpeg_bytes(fig_slope_neg),
        file_name=f"bar_Slope_NEG_{assay_pick}_{ch_pick}.jpeg".replace(" ", "_"),
        mime="image/jpeg"
    )

st.success("Pronto ✅")