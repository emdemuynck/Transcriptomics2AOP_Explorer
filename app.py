
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from t2aop_stats import summarize_deg_totals, summarize_magnitude, summarize_css_pairs

from t2aop_engine import (
    validate_deseq, validate_mapping, build_sample_dict,
    compute_cosine_similarity_by_set,
    circular_barplot_geneset, circular_panel_grid, clean_sample_label,
    flat_barplot_geneset, profile_panel_grid, dual_flat_comparison,
    deg_summary_for_samples, plot_deg_grouped_bars,
    plot_triangular_similarity_clean,
    median_css_all_pairs,
    silhouette_from_similarity, permutation_test_silhouette, exact_silhouette_permutations,
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

EXAMPLE_DIR = Path(__file__).parent / "examples"
EXAMPLE_DESEQ = EXAMPLE_DIR / "example_deseq2_results.csv"
EXAMPLE_MAPPING = EXAMPLE_DIR / "example_gene_set_mapping.csv"
HAS_EXAMPLES = EXAMPLE_DESEQ.exists() and EXAMPLE_MAPPING.exists()

if HAS_EXAMPLES:
    st.sidebar.button(
        "Load example data",
        help="Loads a small synthetic dataset so the interface can be explored "
             "without uploading files.",
        on_click=lambda: st.session_state.update(use_example=True),
    )
    if st.session_state.get("use_example") and not (deseq_file or mapping_file):
        st.sidebar.caption("Example data loaded. Upload your own files to replace it.")


@st.cache_data
def load_example():
    return pd.read_csv(EXAMPLE_DESEQ), pd.read_csv(EXAMPLE_MAPPING)


using_example = bool(st.session_state.get("use_example")) and not (deseq_file or mapping_file)

if not (deseq_file and mapping_file) and not using_example:
    st.info(
        "Upload a combined DESeq2 table and a Transcriptomics2AOP mapping table to start, "
        "or use **Load example data** in the sidebar to explore the interface."
    )
    st.stop()

try:
    if using_example:
        raw_deseq, raw_mapping = load_example()
    else:
        raw_deseq = load_table(deseq_file)
        raw_mapping = load_table(mapping_file)
except Exception as e:
    st.error(str(e))
    st.stop()

if using_example:
    st.warning(
        "Showing a small synthetic dataset provided for demonstration. "
        "It carries no biological meaning and is not the data analysed in the publication.",
        icon="⚠️",
    )

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

st.sidebar.header("3b. Sample labels")

auto_clean_labels = st.sidebar.checkbox(
    "Clean sample labels automatically",
    value=True,
    help=(
        "Strips the trailing .csv and the solvent/control suffix "
        "(e.g. 'MMS_C5_DMSO E.csv' becomes 'MMS C5'). Labels are only "
        "changed for display; the underlying sample names are untouched."
    )
)

if "label_overrides" not in st.session_state:
    st.session_state["label_overrides"] = {}

with st.sidebar.expander("Rename samples manually"):
    st.caption(
        "Leave a field empty to keep the automatic label. Overrides apply "
        "to every figure and are kept for this session."
    )
    label_table = pd.DataFrame({
        "sample": selected_samples,
        "automatic label": [
            clean_sample_label(s) if auto_clean_labels else s
            for s in selected_samples
        ],
        "custom label": [
            st.session_state["label_overrides"].get(s, "")
            for s in selected_samples
        ],
    })
    edited_labels = st.data_editor(
        label_table,
        hide_index=True,
        use_container_width=True,
        disabled=["sample", "automatic label"],
        key="label_editor"
    )
    st.session_state["label_overrides"] = {
        row["sample"]: str(row["custom label"]).strip()
        for _, row in edited_labels.iterrows()
        if str(row["custom label"]).strip()
    }

label_overrides = st.session_state["label_overrides"]


def display_label(sample):
    """Label used in figures and download names."""
    if sample in label_overrides and label_overrides[sample]:
        return label_overrides[sample]
    return clean_sample_label(sample) if auto_clean_labels else str(sample)


profile_style = st.sidebar.radio(
    "Profile style",
    options=["Circular", "Flat (aligned)"],
    index=0,
    horizontal=True,
    key="profile_style",
    help=(
        "Circular profiles give an immediate impression of how broadly and how "
        "strongly a gene set responds. Flat profiles keep each gene at the same "
        "position on the x axis, so the same gene can be compared across samples "
        "column by column, which is easier to read when several profiles are "
        "shown side by side."
    )
)
STYLE = "flat" if profile_style.startswith("Flat") else "circular"
profile_fn = flat_barplot_geneset if STYLE == "flat" else circular_barplot_geneset

with st.sidebar.expander("Profile colours"):
    st.caption(
        "Applies to every circular profile. Keep these consistent with the "
        "colour conventions used in the rest of the manuscript."
    )
    up_color = st.color_picker("Significantly upregulated", "#E69F00", key="col_up")
    down_color = st.color_picker("Significantly downregulated", "#0072B2", key="col_down")
    nonsig_color = st.color_picker("Not significant", "#008000", key="col_ns")
    missing_color = st.color_picker("Not detected in sample", "#8A9099", key="col_missing")

profile_colors = dict(
    up_color=up_color,
    down_color=down_color,
    nonsig_color=nonsig_color,
    missing_color=missing_color
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
    value=5.0,
    step=1.0,
    help="Default = 5, as used in Demuynck et al. The weight -log10(padj) is clipped at this value."
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

# ----------------------------------------------------------------
# Apply the display labels once, here, so that every downstream
# visual and table (circular profiles, DEG counts, magnitude,
# similarity matrices, clustering, exports) uses the same names.
# ----------------------------------------------------------------
_label_map, _seen = {}, {}
for _raw in selected_samples:
    _lab = display_label(_raw)
    if _lab in _seen and _seen[_lab] != _raw:
        # two different samples would collide: keep them distinguishable
        _n = sum(1 for v in _label_map.values() if v == _lab or v.startswith(_lab + " ("))
        _lab = f"{_lab} ({_n + 1})"
    _seen[_lab] = _raw
    _label_map[_raw] = _lab

if any(k != v for k, v in _label_map.items()):
    deseq_sel["sample"] = deseq_sel["sample"].map(_label_map).fillna(deseq_sel["sample"])
    selected_samples = [_label_map[s] for s in selected_samples]

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
    "📈 log2FC magnitude",
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
            "Choose whether to display genes shared by all selected samples "
            "or genes observed in at least one selected sample."
        )
    )
    circular_sets = st.multiselect(
        "Gene sets to plot",
        selected_sets,
        default=selected_sets[:1],
        key="circular_sets"
    )

    circular_gene_mode = st.radio(
        "Genes to display across selected samples",
        options=["Common genes (intersection)", "All observed genes (union)"],
        index=0,
        horizontal=True,
        key="circular_gene_mode",
        help=(
            "Intersection: only genes found in every selected sample. "
            "Union: genes found in at least one selected sample; "
            "missing genes are indicated in grey, not treated as measured non-DEGs."
        )
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

    genes_by_set = {}

    for set_id in circular_sets:
        # Start from all genes assigned to the selected T2AOP gene set
        geneset_genes = set(
            mapping.loc[
                mapping["gene_set"] == set_id, "gene_symbol"
            ].dropna().astype(str).str.upper().unique().tolist()
        )

        if not circular_samples:
            st.markdown(f"### {set_id}")
            st.info("Select at least one sample to generate circular profiles.")
            continue

        sample_gene_sets = []
        for selected_sample in circular_samples:
            sample_df = sample_dict[selected_sample]
            observed = sample_df.loc[
                pd.to_numeric(sample_df["log2FoldChange"], errors="coerce").notna(),
                "gene_symbol"
            ].dropna().astype(str).str.upper()
            sample_gene_sets.append(set(observed) & geneset_genes)

        if circular_gene_mode.startswith("Common"):
            displayed_genes = set.intersection(*sample_gene_sets)
            mode_label = "shared by all selected samples"
        else:
            displayed_genes = set.union(*sample_gene_sets)
            mode_label = "observed in at least one selected sample"

        genes = sorted(displayed_genes)
        genes_by_set[set_id] = genes
        st.markdown(f"### {set_id}")
        st.caption(
            f"{len(genes)} genes {mode_label} "
            f"(out of {len(geneset_genes)} mapped genes) "
            f"across {len(circular_samples)} selected sample(s)."
        )
        if circular_gene_mode.startswith("All"):
            st.caption(
                "Grey gene labels indicate genes absent from a sample (or without a "
                "usable log2FC). These are missing values, not measured non-DEGs."
            )

        if len(genes) < 3:
            st.warning(f"{set_id}: fewer than 3 genes available for a circular profile.")
            continue

        for sample in circular_samples:
            try:
                fig = profile_fn(
                    sample_dict[sample],
                    genes=genes,
                    title="",
                    sample_name=sample,
                    gene_set=set_id,
                    padj_thr=padj_thr,
                    alphabetical=alphabetical,
                    top_n=None if top_n == 0 else top_n,
                    clip_max=clip_max,
                    show_gene_labels=show_gene_labels,
                    **profile_colors
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

    # ------------------------------------------------------------
    # Multi-panel figure
    # ------------------------------------------------------------
    st.divider()
    st.markdown("### Multi-panel figure")
    st.caption(
        "Assemble the selected samples and gene sets into a single figure, "
        "with gene sets as rows and samples as columns. All circles share the "
        "same radial scale, so bar lengths are comparable across the panel."
    )

    panel_sets = [s for s in circular_sets if len(genes_by_set.get(s, [])) >= 3]

    if not circular_samples or not panel_sets:
        st.info("Select at least one sample and one gene set with three or more genes.")
    else:
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            panel_size = st.slider(
                "Panel size (inches)", min_value=2.0, max_value=8.0,
                value=4.0, step=0.2, key="panel_size"
            )
        with pc2:
            panel_dpi = st.select_slider(
                "Resolution (dpi)", options=[150, 200, 300, 600],
                value=300, key="panel_dpi"
            )
        with pc3:
            panel_gene_labels = st.checkbox(
                "Show gene symbols in panel", value=False, key="panel_gene_labels",
                help="Usually off: with many genes the symbols are unreadable at panel size."
            )

        st.caption(
            f"{len(panel_sets)} row(s) x {len(circular_samples)} column(s). "
            "Row order follows the gene-set selection, column order the sample selection."
        )

        if st.button("Build multi-panel figure", key="build_panel"):
            with st.spinner("Assembling panel..."):
                try:
                    fig_panel = profile_panel_grid(
                        sample_dict,
                        genes_by_set=genes_by_set,
                        samples=circular_samples,
                        gene_sets=panel_sets,
                        style=STYLE,
                        padj_thr=padj_thr,
                        alphabetical=alphabetical,
                        clip_max=clip_max,
                        show_gene_labels=panel_gene_labels,
                        panel_size=panel_size,
                        dpi=panel_dpi,
                        **profile_colors
                    )
                    st.pyplot(fig_panel, clear_figure=False)
                    st.download_button(
                        "Download multi-panel figure (PNG)",
                        data=fig_to_png_bytes(fig_panel),
                        file_name="T2AOP_panel_{}.png".format("_".join(panel_sets)),
                        mime="image/png",
                        key="panel_dl"
                    )
                    plt.close(fig_panel)
                except Exception as e:
                    st.error(f"Could not build the panel: {e}")


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
            help="A gene is highlighted when padj is below the selected threshold in both samples and both log2FC values have the same sign. "
                 "In the mirrored (flat) comparison this is replaced by the concordance shading below."
        )
        mark_conc = st.checkbox(
            "Shade concordant / discordant genes (mirrored comparison)",
            value=True,
            help="Every gene gets a band behind the bars according to the sign of its two "
                 "log2FC values, whether or not it reaches significance: one tint for the same "
                 "direction, another tint plus a hatch for opposite directions. Gene symbols are "
                 "printed in bold purple only for the genes that are significant in both samples."
        )
        compare_labels = st.checkbox("Show gene symbols in comparison", value=True)
        order_label = st.radio(
            "Gene order (mirrored comparison)",
            ["Alphabetical", "By log2FC of sample A", "By concordance (A x B)"],
            index=0, horizontal=True, key="compare_order",
            help="Both panels always use the same order, so a gene sits at the "
                 "same position top and bottom. Alphabetical makes a named gene "
                 "easy to find; the other two group the strongest or the most "
                 "concordant responses together."
        )
        ORDER_BY = {"Alphabetical": "alphabetical",
                    "By log2FC of sample A": "magnitude",
                    "By concordance (A x B)": "concordance"}[order_label]

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

            if STYLE == "flat":
                fig_dual = dual_flat_comparison(
                    sample_dict[compare_a], sample_dict[compare_b],
                    genes=common, name_a=display_label(compare_a),
                    name_b=display_label(compare_b), gene_set=compare_set,
                    padj_thr=padj_thr, clip_max=3,
                    show_gene_labels=compare_labels,
                    highlight_genes=(shared_same if mark_conc or highlight_shared else []),
                    mark_concordance=mark_conc,
                    order_by=ORDER_BY,
                    **profile_colors
                )
                st.pyplot(fig_dual, clear_figure=False)
                st.caption(
                    "Mirrored profiles: genes keep the same position in both panels and "
                    "the lower panel is inverted, so a gene regulated in the same direction "
                    "in both samples appears as two bars pointing away from the centre."
                )
                st.download_button(
                    "Download dual comparison",
                    fig_to_png_bytes(fig_dual),
                    file_name=f"{compare_a}_vs_{compare_b}_{compare_set}_dual.png",
                    mime="image/png", key=f"dual_dl_{compare_set}"
                )
                plt.close(fig_dual)
                figs = []
            else:
                figs = []
                cols = st.columns(2)
                for col, sample in zip(cols, [compare_a, compare_b]):
                    with col:
                        fig_cmp = circular_barplot_geneset(
                            sample_dict[sample], genes=common, title="",
                            sample_name=sample, gene_set=compare_set,
                            padj_thr=padj_thr, alphabetical=True,
                            top_n=None, clip_max=3, show_gene_labels=compare_labels,
                            highlight_genes=shared_same if highlight_shared else None,
                            **profile_colors
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

    st.markdown("### Total DEGs per sample (whole DESeq2 table)")
    st.caption("Counts are independent of the selected gene sets. Unique genes with non-missing padj and log2FC are tested; padj ≤ threshold and |log2FC| > threshold.")
    total_degs = summarize_deg_totals(sample_dict, padj_thr, fc_thr)
    st.dataframe(total_degs, use_container_width=True)
    st.download_button("Download total DEG counts", total_degs.to_csv(index=False).encode("utf-8"),
                       file_name="T2AOP_total_DEGs_by_sample.csv", mime="text/csv")
    st.markdown("### DEG counts within a selected gene set")

    dc1, dc2 = st.columns([1, 1.4])
    with dc1:
        deg_set = st.selectbox(
            "Gene set / branch",
            selected_sets,
            key="deg_set"
        )
    with dc2:
        deg_common_only = st.checkbox(
            "Restrict to genes detected in all selected samples",
            value=False,
            key="deg_common_only",
            help=(
                "Samples profiled on different TempO-Seq panels cover different "
                "numbers of genes in a gene set, so the percentages are computed "
                "on different denominators and are not comparable between samples. "
                "Restricting to the shared genes applies the same denominator to "
                "every sample, as is already done for the cosine similarity scores."
            )
        )

    genes = mapping.loc[
        mapping["gene_set"] == deg_set, "gene_symbol"
    ].dropna().astype(str).str.upper().unique().tolist()

    n_mapped = len(genes)
    if deg_common_only:
        detected = [
            set(
                df.loc[
                    pd.to_numeric(df["log2FoldChange"], errors="coerce").notna(),
                    "gene_symbol"
                ].dropna().astype(str).str.upper()
            )
            for df in sample_dict.values()
        ]
        shared = set.intersection(*detected) & set(genes) if detected else set()
        genes = sorted(shared)
        if len(genes) < 3:
            st.error(
                f"Only {len(genes)} of the {n_mapped} {deg_set} genes are detected in "
                "every selected sample. Deselect the samples profiled on a different "
                "panel, or uncheck the option."
            )
            st.stop()
        st.caption(
            f"Restricted to the {len(genes)} of {n_mapped} {deg_set} genes detected in "
            f"all {len(sample_dict)} selected samples; percentages therefore share the "
            "same denominator."
        )
    else:
        st.caption(
            f"Percentages are computed per sample on the {deg_set} genes detected in "
            "that sample, so denominators differ between samples profiled on different "
            "TempO-Seq panels (see the n_genes_found column)."
        )

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
            file_name=f"{deg_set}_DEG_summary{'_shared_genes' if deg_common_only else ''}.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download DEG figure",
            data=deg_png,
            file_name=f"{deg_set}_DEG_counts{'_shared_genes' if deg_common_only else ''}.png",
            mime="image/png"
        )
    plt.close(fig)


# ================================================================
# LOG2FC MAGNITUDE
# ================================================================
with tabs[4]:
    st.subheader("Gene-set log2FC magnitude statistics")
    st.caption("Descriptive log2FC summaries; these are not enrichment scores or evidence that a key event occurred.")
    sig_only = st.checkbox("Use only DEGs (padj and |log2FC| thresholds)", value=False,
                           help="Otherwise all genes with a finite log2FC in the gene set are used.")
    magnitude = summarize_magnitude(sample_dict, mapping_sel, selected_sets,
                                    padj_thr=padj_thr, fc_thr=fc_thr,
                                    significant_only=sig_only)
    st.dataframe(magnitude.round(4), use_container_width=True)
    metric = st.selectbox("Magnitude statistic to plot", ["mean_abs_log2fc", "median_abs_log2fc",
        "p90_abs_log2fc", "rms_log2fc", "mean_log2fc", "median_log2fc", "max_abs_log2fc"])
    if not magnitude.empty:
        chart = magnitude.pivot(index="sample", columns="gene_set", values=metric)
        st.bar_chart(chart)
    st.download_button("Download log2FC magnitude statistics",
                       magnitude.to_csv(index=False).encode("utf-8"),
                       file_name="T2AOP_log2FC_magnitude_by_sample_and_set.csv", mime="text/csv")


# ================================================================
# SIMILARITY
# ================================================================
with tabs[5]:
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

            st.markdown("#### Per-compound-pair CSS statistics")
            st.caption("Each compound pair aggregates all cross-compound sample pairs (upper triangle only); self-comparisons and mirrored duplicates are excluded. P90 is calculated separately for each compound pair, not globally.")
            css_summary, css_raw = summarize_css_pairs(sim, separator=separator,
                chemical_level=int(chemical_level))
            if css_summary.empty:
                st.info("No between-compound comparisons for the selected samples.")
            else:
                st.dataframe(css_summary.round(4), use_container_width=True)
                st.download_button("Download per-compound CSS statistics",
                    css_summary.to_csv(index=False).encode("utf-8"),
                    file_name=f"{similarity_set}_CSS_pair_statistics.csv", mime="text/csv")
                st.download_button("Download underlying sample-pair CSS values",
                    css_raw.to_csv(index=False).encode("utf-8"),
                    file_name=f"{similarity_set}_CSS_sample_pairs.csv", mime="text/csv")
                selected_pair = st.selectbox("Inspect compound pair",
                    css_summary.apply(lambda r: f"{r['compound_A']} — {r['compound_B']}", axis=1).tolist())
                chem_a, chem_b = selected_pair.split(" — ", 1)
                values = css_raw.loc[(css_raw.compound_A == chem_a) &
                                     (css_raw.compound_B == chem_b), "CSS"]
                fig_dist, ax_dist = plt.subplots(figsize=(7, 3.3))
                ax_dist.hist(values, bins=min(20, max(3, len(values))), color="#6387B5", edgecolor="white")
                ax_dist.axvline(np.percentile(values, 90), color="#BD4D5A", linestyle="--", label="P90")
                ax_dist.axvline(values.max(), color="#3B7660", linestyle=":", label="Maximum")
                ax_dist.set_xlabel("Cosine similarity")
                ax_dist.set_ylabel("Number of sample comparisons")
                ax_dist.legend(frameon=False)
                fig_dist.tight_layout()
                st.pyplot(fig_dist)
                plt.close(fig_dist)

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
with tabs[6]:
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

                st.markdown("#### Exact compound-level permutation test")
                st.caption("Enumerates every distinct assignment of compounds to predefined groups, preserving group sizes (e.g. 8 compounds in groups of 3 and 5 = 56 assignments). All concentrations of each compound stay together. Silhouettes are computed on the same sample-level distance matrix as above.")
                if st.button("Run exact exhaustive test", key="run_exact_silhouette"):
                    with st.spinner("Enumerating all group assignments..."):
                        exact_result = exact_silhouette_permutations(
                            sim, groups, separator=separator, chemical_level=int(chemical_level))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Observed silhouette", f"{exact_result['observed']:.3f}")
                    c2.metric("Exact one-sided p-value", f"{exact_result['p_value']:.5f}")
                    c3.metric("Assignments tested", str(exact_result['n_assignments']))
                    st.caption(f"{exact_result['n_extreme']} / {exact_result['n_assignments']} assignments have silhouette ≥ observed. Observed assignment included; no Monte Carlo correction.")
                    fig_exact, ax_exact = plt.subplots(figsize=(7, 3.5))
                    ax_exact.hist(exact_result['distribution']['silhouette_score'], bins=16, edgecolor='white')
                    ax_exact.axvline(exact_result['observed'], color='crimson', linestyle='--', linewidth=2, label='Observed')
                    ax_exact.set_xlabel('Silhouette score (all compound assignments)')
                    ax_exact.set_ylabel('Number of assignments')
                    ax_exact.legend()
                    st.pyplot(fig_exact)
                    plt.close(fig_exact)
                    st.dataframe(exact_result['distribution'], use_container_width=True)
                    st.download_button("Download all exact assignments (CSV)",
                        exact_result['distribution'].to_csv(index=False).encode('utf-8'),
                        file_name=f"{subgroup_set}_exact_silhouette_assignments.csv", mime="text/csv")
                    st.caption("Group labels must be defined independently of the similarity matrix for valid inference. This test measures separation of the predefined groups, not validation of a genotoxic mechanism.")

                st.markdown("#### Chemical-level silhouette permutation test")
                st.caption("Null hypothesis: the observed grouping does not separate transcriptional profiles better than random assignments of whole chemicals to groups. Group sizes (number of chemicals) are preserved; sample profiles and concentrations remain together.")
                n_perm = st.number_input("Requested permutations", min_value=99, max_value=19999,
                                         value=999, step=100, key="silhouette_n_permutations")
                seed = st.number_input("Random seed", min_value=0, max_value=999999,
                                       value=42, step=1, key="silhouette_seed")
                if st.button("Run silhouette permutation test", key="run_silhouette_permutations"):
                    with st.spinner("Computing null silhouette distribution..."):
                        result = permutation_test_silhouette(
                            sim, groups, separator=separator, chemical_level=int(chemical_level),
                            n_permutations=int(n_perm), random_state=int(seed))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Observed silhouette", f"{result['observed']:.3f}")
                    m2.metric("One-sided permutation p-value", f"{result['p_value']:.4g}")
                    m3.metric("Null mean silhouette", f"{result['null_mean']:.3f}")
                    st.caption(f"{result['n_permutations']} distinct permutations; "
                               f"{result['n_chemicals']} chemicals / {result['n_samples']} samples. "
                               + ("Exact permutation test." if result['exact'] else "Monte Carlo permutation test with +1 correction."))
                    fig_perm, ax_perm = plt.subplots(figsize=(7, 3.5))
                    ax_perm.hist(result['null_scores'], bins=min(30, max(5, len(result['null_scores'])//5)),
                                 edgecolor='white', alpha=.85)
                    ax_perm.axvline(result['observed'], color='crimson', linestyle='--', linewidth=2,
                                    label=f"Observed = {result['observed']:.3f}")
                    ax_perm.set_xlabel("Silhouette score under permuted chemical groups")
                    ax_perm.set_ylabel("Number of permutations")
                    ax_perm.legend()
                    st.pyplot(fig_perm)
                    plt.close(fig_perm)
                    perm_df = pd.DataFrame({'permutation': np.arange(1, result['n_permutations']+1),
                                            'silhouette_score': result['null_scores']})
                    st.download_button("Download permutation distribution (CSV)",
                                       data=perm_df.to_csv(index=False).encode('utf-8'),
                                       file_name=f"{subgroup_set}_silhouette_permutations.csv", mime="text/csv")
                    summary = pd.DataFrame([{k:v for k,v in result.items() if k != 'null_scores'}])
                    st.download_button("Download permutation test summary (CSV)",
                                       data=summary.to_csv(index=False).encode('utf-8'),
                                       file_name=f"{subgroup_set}_silhouette_permutation_summary.csv", mime="text/csv")
                    st.caption("A small p-value supports non-random separation of the predefined groups in this dataset; it does not validate mechanistic classification. Groups should be defined independently of the tested CSS matrix.")

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
with tabs[7]:
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
