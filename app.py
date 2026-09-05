
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from t2aop_engine import (
    validate_deseq, validate_mapping, build_sample_dict,
    compute_cosine_similarity_by_set,
    circular_barplot_geneset,
    deg_summary_for_samples, plot_deg_grouped_bars,
    plot_triangular_similarity_clean,
    median_css_all_pairs,
    silhouette_from_similarity,
    extract_chemical
)

st.set_page_config(
    page_title="Transcriptomics2AOP Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: #F7F9FC; }
    [data-testid="stSidebar"] { background: #EEF2F7; border-right: 1px solid #DCE3EC; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { color: #172033; letter-spacing: -0.02em; }
    h1 { font-weight: 750; }
    [data-testid="stMetric"] { background: white; border: 1px solid #E3E8EF; border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(23,32,51,.04); }
    [data-testid="stDataFrame"] { border: 1px solid #E3E8EF; border-radius: 12px; overflow: hidden; }
    .stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid #D8DEE8; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid #E3E8EF; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding-left: 1rem; padding-right: 1rem; }
    div[data-testid="stFileUploader"] { background: white; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Transcriptomics2AOP Explorer")
st.caption(
    "A mechanistically anchored interface for DESeq2-based Transcriptomics2AOP analyses."
)

@st.cache_data
def load_table(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type")


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def parse_groups(text):
    groups = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, vals = line.split(":", 1)
        groups[name.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    return groups


# ================================================================
# SIDEBAR — INPUT
# ================================================================
st.sidebar.header("1. Input files")

deseq_file = st.sidebar.file_uploader(
    "Combined DESeq2 results",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    help="One row per gene/sample. A sample column is required."
)

mapping_file = st.sidebar.file_uploader(
    "T2AOP gene-set mapping",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    help="At minimum: gene symbol + gene-set/KE/branch identifier."
)

if not (deseq_file and mapping_file):
    st.info(
        "Upload a combined DESeq2 table and a Transcriptomics2AOP mapping table to start."
    )
    st.stop()

try:
    raw_deseq = load_table(deseq_file)
    raw_mapping = load_table(mapping_file)
except Exception as e:
    st.error(str(e))
    st.stop()

# ================================================================
# COLUMN MAPPING
# ================================================================
st.sidebar.header("2. Column mapping")

def find_default(cols, candidates, fallback=0):
    lower = [str(c).lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(lower):
            if cand == c or cand in c:
                return i
    return min(fallback, len(cols)-1)

gene_col = st.sidebar.selectbox(
    "DESeq2 gene symbol",
    raw_deseq.columns,
    index=find_default(raw_deseq.columns, ["gene_symbol", "gene"])
)
fc_col = st.sidebar.selectbox(
    "log2 fold change",
    raw_deseq.columns,
    index=find_default(raw_deseq.columns, ["log2foldchange", "log2fc"], 1)
)
padj_col = st.sidebar.selectbox(
    "Adjusted p-value",
    raw_deseq.columns,
    index=find_default(raw_deseq.columns, ["padj", "adjust"], 2)
)
sample_col = st.sidebar.selectbox(
    "Sample",
    raw_deseq.columns,
    index=find_default(raw_deseq.columns, ["sample", "condition", "source_file"], 3)
)

map_gene_col = st.sidebar.selectbox(
    "Mapping gene symbol",
    raw_mapping.columns,
    index=find_default(raw_mapping.columns, ["gene_symbol", "gene"])
)
map_set_col = st.sidebar.selectbox(
    "Mapping gene set / KE / branch",
    raw_mapping.columns,
    index=find_default(raw_mapping.columns, ["ke_id_full", "ke_id", "geneset", "gene_set"], 1)
)

try:
    deseq = validate_deseq(raw_deseq, gene_col, fc_col, padj_col, sample_col)
    mapping = validate_mapping(raw_mapping, map_gene_col, map_set_col)
except Exception as e:
    st.error(str(e))
    st.stop()

# standardize internal names
deseq = deseq.rename(columns={
    gene_col: "gene_symbol",
    fc_col: "log2FoldChange",
    padj_col: "padj",
    sample_col: "sample"
})
mapping = mapping.rename(columns={
    map_gene_col: "gene_symbol",
    map_set_col: "gene_set"
})

samples_all = sorted(deseq["sample"].unique().tolist())
sets_all = sorted(mapping["gene_set"].unique().tolist())

# ================================================================
# ANALYSIS OPTIONS
# ================================================================
st.sidebar.header("3. Analysis selection")

selected_samples = st.sidebar.multiselect(
    "Samples",
    samples_all,
    default=samples_all[:min(12, len(samples_all))]
)

selected_sets = st.sidebar.multiselect(
    "Gene sets / branches / biomarkers",
    sets_all,
    default=sets_all[:1]
)

st.sidebar.header("4. T2AOP parameters")
padj_thr = st.sidebar.number_input(
    "DEG padj threshold",
    min_value=0.0001,
    max_value=1.0,
    value=0.05,
    format="%.4f"
)
fc_thr = st.sidebar.number_input(
    "Absolute log2FC threshold for DEG counts",
    min_value=0.0,
    value=0.0,
    step=0.1
)
weighted = st.sidebar.checkbox(
    "Weight CSS by -log10(padj)",
    value=True
)
max_weight = st.sidebar.number_input(
    "Maximum -log10(padj) weight",
    min_value=1.0,
    value=10.0,
    step=1.0,
    help="Notebook default = 10"
)
common_genes_only = st.sidebar.checkbox(
    "Use only genes present in all selected samples",
    value=True
)
linkage_method = st.sidebar.selectbox(
    "Hierarchical clustering linkage",
    ["average", "complete", "single", "weighted"],
    index=0
)

st.sidebar.header("5. Chemical parsing")
separator = st.sidebar.text_input("Sample separator", value="_")
chemical_level = st.sidebar.number_input(
    "Chemical name parts",
    min_value=1,
    value=1,
    step=1
)

if len(selected_samples) < 1 or len(selected_sets) < 1:
    st.warning("Select at least one sample and one gene set.")
    st.stop()

deseq_sel = deseq[deseq["sample"].isin(selected_samples)].copy()
sample_dict = build_sample_dict(deseq_sel, "sample")

mapping_sel = mapping[mapping["gene_set"].isin(selected_sets)].copy()

# ================================================================
# NAVIGATION
# ================================================================
tabs = st.tabs([
    "📋 Overview",
    "🧬 Circular profiles",
    "⚖️ Compare profiles",
    "📊 DEG counts",
    "🔗 Similarity",
    "🧭 Subgrouping",
    "🩺 Diagnostics"
])

# ================================================================
# OVERVIEW
# ================================================================
with tabs[0]:
    st.subheader("Input overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(selected_samples))
    c2.metric("Selected gene sets", len(selected_sets))
    c3.metric("DESeq2 rows", len(deseq_sel))

    sample_summary = (
        deseq_sel.groupby("sample")["gene_symbol"]
        .nunique()
        .reset_index(name="n_unique_genes")
    )
    set_summary = (
        mapping_sel.groupby("gene_set")["gene_symbol"]
        .nunique()
        .reset_index(name="n_genes")
        .sort_values("n_genes", ascending=False)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Samples")
        st.dataframe(sample_summary, use_container_width=True)
    with right:
        st.markdown("#### Gene sets")
        st.dataframe(set_summary, use_container_width=True)

    with st.expander("Preview combined DESeq2 table"):
        st.dataframe(deseq_sel.head(100), use_container_width=True)

    with st.expander("Preview T2AOP mapping"):
        st.dataframe(mapping_sel.head(100), use_container_width=True)


# ================================================================
# CIRCULAR PROFILES
# ================================================================
with tabs[1]:
    st.subheader("T2AOP circular gene-expression profiles")

    circular_samples = st.multiselect(
        "Samples to plot",
        selected_samples,
        default=selected_samples[:min(4, len(selected_samples))],
        key="circular_samples",
        help=(
            "Circular profiles use only gene-set genes that are present in "
            "ALL samples selected here, ensuring directly comparable profiles."
        )
    )
    circular_sets = st.multiselect(
        "Gene sets to plot",
        selected_sets,
        default=selected_sets[:1],
        key="circular_sets"
    )

    alphabetical = st.checkbox("Alphabetical gene order", value=True)
    show_gene_labels = st.checkbox(
        "Show gene symbols",
        value=True,
        help="Display gene symbols around the circular profile. Disable for a cleaner overview when many genes are plotted."
    )
    top_n = st.number_input(
        "Maximum genes per plot (0 = all)",
        min_value=0,
        value=0,
        step=10
    )
    clip_max = st.number_input(
        "Clip displayed log2FC at ±",
        min_value=0.5,
        value=3.0,
        step=0.5
    )

    for set_id in circular_sets:
        # Start from all genes assigned to the selected T2AOP gene set
        geneset_genes = set(
            mapping.loc[
                mapping["gene_set"] == set_id, "gene_symbol"
            ].dropna().astype(str).str.upper().unique().tolist()
        )

        # Keep ONLY genes that are present in every sample selected
        # in the Circular profiles tab.
        common_genes = set(geneset_genes)

        for selected_sample in circular_samples:
            sample_genes = set(
                sample_dict[selected_sample]["gene_symbol"]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )
            common_genes &= sample_genes

        genes = sorted(common_genes)

        st.markdown(f"### {set_id}")
        st.caption(
            f"{len(genes)} common genes displayed "
            f"(out of {len(geneset_genes)} genes in the gene set) "
            f"across {len(circular_samples)} selected sample(s)."
        )

        if len(circular_samples) == 0:
            st.info("Select at least one sample to generate circular profiles.")
            continue

        if len(genes) < 3:
            st.warning(
                f"{set_id}: fewer than 3 gene-set genes are shared by all "
                "selected samples, so a circular profile cannot be generated."
            )
            continue

        for sample in circular_samples:
            try:
                fig = circular_barplot_geneset(
                    sample_dict[sample],
                    genes=genes,
                    title="",
                    sample_name=sample,
                    gene_set=set_id,
                    padj_thr=padj_thr,
                    alphabetical=alphabetical,
                    top_n=None if top_n == 0 else top_n,
                    clip_max=clip_max,
                    show_gene_labels=show_gene_labels
                )
                png_bytes = fig_to_png_bytes(fig)
                st.pyplot(fig, clear_figure=False)

                st.download_button(
                    f"Download {sample} — {set_id}",
                    data=png_bytes,
                    file_name=f"{sample}_{set_id}_circular.png",
                    mime="image/png",
                    key=f"circ_{sample}_{set_id}"
                )
                plt.close(fig)
            except Exception as e:
                st.warning(f"{sample} / {set_id}: {e}")


# ================================================================
# COMPARE CIRCULAR PROFILES
# ================================================================
with tabs[2]:
    st.subheader("Side-by-side T2AOP profile comparison")
    st.caption(
        "Compare two samples using the same gene universe. Genes significantly "
        "perturbed in the same direction in both samples can be highlighted."
    )

    if len(selected_samples) < 2:
        st.info("Select at least two samples in the sidebar.")
    else:
        cc1, cc2, cc3 = st.columns([1, 1, 1])
        with cc1:
            compare_a = st.selectbox("Sample A", selected_samples, index=0, key="compare_a")
        with cc2:
            compare_b = st.selectbox("Sample B", selected_samples, index=1, key="compare_b")
        with cc3:
            compare_set = st.selectbox("Gene set / branch", selected_sets, key="compare_set")

        highlight_shared = st.checkbox(
            "Highlight significant genes changing in the same direction",
            value=True,
            help="A gene is highlighted when padj is below the selected threshold in both samples and both log2FC values have the same sign."
        )
        compare_labels = st.checkbox("Show gene symbols in comparison", value=True)

        if compare_a == compare_b:
            st.warning("Choose two different samples.")
        else:
            geneset_genes = set(
                mapping.loc[mapping["gene_set"] == compare_set, "gene_symbol"]
                .dropna().astype(str).str.upper().unique()
            )
            da = sample_dict[compare_a].copy()
            db = sample_dict[compare_b].copy()
            da["gene_symbol"] = da["gene_symbol"].astype(str).str.upper()
            db["gene_symbol"] = db["gene_symbol"].astype(str).str.upper()
            common = sorted(
                geneset_genes
                & set(da["gene_symbol"].dropna())
                & set(db["gene_symbol"].dropna())
            )

            a = da[da["gene_symbol"].isin(common)][["gene_symbol", "log2FoldChange", "padj"]].drop_duplicates("gene_symbol")
            b = db[db["gene_symbol"].isin(common)][["gene_symbol", "log2FoldChange", "padj"]].drop_duplicates("gene_symbol")
            pair = a.merge(b, on="gene_symbol", suffixes=("_A", "_B"))
            shared_same = pair.loc[
                (pair["padj_A"] <= padj_thr)
                & (pair["padj_B"] <= padj_thr)
                & (np.sign(pair["log2FoldChange_A"]) == np.sign(pair["log2FoldChange_B"]))
                & (pair["log2FoldChange_A"] != 0),
                "gene_symbol"
            ].tolist()

            st.caption(
                f"{len(common)} common genes in {compare_set}; "
                f"{len(shared_same)} are significantly perturbed in the same direction in both samples."
            )

            figs = []
            cols = st.columns(2)
            for col, sample in zip(cols, [compare_a, compare_b]):
                with col:
                    fig_cmp = circular_barplot_geneset(
                        sample_dict[sample], genes=common, title="",
                        sample_name=sample, gene_set=compare_set,
                        padj_thr=padj_thr, alphabetical=True,
                        top_n=None, clip_max=3, show_gene_labels=compare_labels,
                        highlight_genes=shared_same if highlight_shared else None
                    )
                    png_cmp = fig_to_png_bytes(fig_cmp)
                    st.pyplot(fig_cmp, clear_figure=False)
                    st.download_button(
                        f"Download {sample}", png_cmp,
                        file_name=f"{sample}_{compare_set}_comparison.png",
                        mime="image/png", key=f"cmp_dl_{sample}_{compare_set}"
                    )
                    figs.append(fig_cmp)
            for f in figs:
                plt.close(f)

            if highlight_shared and shared_same:
                st.markdown("**Highlighted shared-direction genes**")
                st.write(", ".join(shared_same))

            with st.expander("Inspect shared-direction gene statistics"):
                st.dataframe(
                    pair[pair["gene_symbol"].isin(shared_same)].sort_values("gene_symbol"),
                    use_container_width=True
                )


# ================================================================
# DEG COUNTS
# ================================================================
with tabs[3]:
    st.subheader("Significantly up- and down-regulated genes")

    deg_set = st.selectbox(
        "Gene set / branch",
        selected_sets,
        key="deg_set"
    )

    genes = mapping.loc[
        mapping["gene_set"] == deg_set, "gene_symbol"
    ].dropna().unique().tolist()

    summary = deg_summary_for_samples(
        sample_dict,
        genes=genes,
        padj_thr=padj_thr,
        fc_thr=fc_thr
    )

    fig = plot_deg_grouped_bars(
        summary,
        padj_thr=padj_thr,
        title=deg_set
    )
    deg_png = fig_to_png_bytes(fig)
    st.pyplot(fig, clear_figure=False)

    st.dataframe(summary, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download DEG summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{deg_set}_DEG_summary.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download DEG figure",
            data=deg_png,
            file_name=f"{deg_set}_DEG_counts.png",
            mime="image/png"
        )
    plt.close(fig)


# ================================================================
# SIMILARITY
# ================================================================
with tabs[4]:
    st.subheader("Weighted cosine similarity and hierarchical clustering")

    if len(selected_samples) < 2:
        st.info("Select at least two samples.")
    else:
        matrices, diagnostics = compute_cosine_similarity_by_set(
            sample_dict,
            mapping_sel.rename(columns={"gene_set": "KE_ID"}),
            set_col="KE_ID",
            gene_col="gene_symbol",
            fc_col="log2FoldChange",
            padj_col="padj",
            weighted=weighted,
            common_genes_only=common_genes_only,
            max_weight=max_weight,
            return_diagnostics=True
        )

        available = [x for x in selected_sets if x in matrices]

        if not available:
            st.warning("No similarity matrix could be generated with the current settings.")
        else:
            similarity_set = st.selectbox(
                "Similarity gene set / branch",
                available,
                key="similarity_set"
            )
            sim = matrices[similarity_set]

            sim_ordered, Z, fig = plot_triangular_similarity_clean(
                sim,
                set_id=similarity_set,
                method=linkage_method
            )
            cluster_png = fig_to_png_bytes(fig)
            st.pyplot(fig, clear_figure=False)

            st.markdown("#### Cosine similarity matrix")
            st.dataframe(sim.round(3), use_container_width=True)

            st.download_button(
                "Download CSS matrix",
                data=sim.to_csv().encode("utf-8"),
                file_name=f"{similarity_set}_CSS_matrix.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download clustered heatmap",
                data=cluster_png,
                file_name=f"{similarity_set}_CSS_clustering.png",
                mime="image/png"
            )
            plt.close(fig)


# ================================================================
# SUBGROUPING
# ================================================================
with tabs[5]:
    st.subheader("Mechanistic subgrouping")

    if len(selected_samples) < 2:
        st.info("Select at least two samples.")
    else:
        matrices, _ = compute_cosine_similarity_by_set(
            sample_dict,
            mapping_sel.rename(columns={"gene_set": "KE_ID"}),
            set_col="KE_ID",
            gene_col="gene_symbol",
            fc_col="log2FoldChange",
            padj_col="padj",
            weighted=weighted,
            common_genes_only=common_genes_only,
            max_weight=max_weight,
            return_diagnostics=True
        )

        available = [x for x in selected_sets if x in matrices]

        if available:
            subgroup_set = st.selectbox(
                "Gene set / branch for subgrouping",
                available,
                key="subgroup_set"
            )
            sim = matrices[subgroup_set]

            st.markdown("### Median CSS between chemicals")
            med = median_css_all_pairs(
                sim,
                separator=separator,
                chemical_level=int(chemical_level)
            )
            st.dataframe(med, use_container_width=True)

            st.download_button(
                "Download median CSS table",
                data=med.to_csv(index=False).encode("utf-8"),
                file_name=f"{subgroup_set}_median_CSS.csv",
                mime="text/csv"
            )

            st.markdown("### Silhouette analysis")
            chemicals = sorted({
                extract_chemical(s, separator, int(chemical_level))
                for s in sim.index
            })

            default_groups = (
                "Group 1: " + ",".join(chemicals[:max(1, len(chemicals)//2)]) +
                "\nGroup 2: " + ",".join(chemicals[max(1, len(chemicals)//2):])
            )

            groups_text = st.text_area(
                "Define expected groups",
                value=default_groups,
                help="One group per line, e.g. Bulky/CisR: AFB1,BAP,CisR"
            )
            groups = parse_groups(groups_text)

            try:
                overall, sample_scores, chem_scores = silhouette_from_similarity(
                    sim,
                    groups,
                    separator=separator,
                    chemical_level=int(chemical_level)
                )

                st.metric("Overall silhouette score", f"{overall:.3f}")

                left, right = st.columns(2)
                with left:
                    st.markdown("#### Per chemical")
                    st.dataframe(chem_scores, use_container_width=True)
                with right:
                    st.markdown("#### Per sample")
                    st.dataframe(sample_scores, use_container_width=True)

                st.download_button(
                    "Download chemical silhouette summary",
                    data=chem_scores.to_csv(index=False).encode("utf-8"),
                    file_name=f"{subgroup_set}_silhouette_by_chemical.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "Download sample silhouette scores",
                    data=sample_scores.to_csv(index=False).encode("utf-8"),
                    file_name=f"{subgroup_set}_silhouette_by_sample.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.info(f"Silhouette analysis: {e}")


# ================================================================
# DIAGNOSTICS
# ================================================================
with tabs[5]:
    st.subheader("Similarity-vector diagnostics")

    if len(selected_samples) >= 2:
        matrices, diagnostics = compute_cosine_similarity_by_set(
            sample_dict,
            mapping_sel.rename(columns={"gene_set": "KE_ID"}),
            set_col="KE_ID",
            gene_col="gene_symbol",
            fc_col="log2FoldChange",
            padj_col="padj",
            weighted=weighted,
            common_genes_only=common_genes_only,
            max_weight=max_weight,
            return_diagnostics=True
        )
        st.dataframe(diagnostics, use_container_width=True)
        st.download_button(
            "Download diagnostics",
            data=diagnostics.to_csv(index=False).encode("utf-8"),
            file_name="T2AOP_similarity_diagnostics.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "Core analytical behavior follows the uploaded Transcriptomics2AOP notebook: "
    "weighted signatures use log2FC × -log10(padj), with a default maximum weight of 10."
)
