
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.gridspec import GridSpec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples


def normalize_gene_symbols(series):
    return series.astype(str).str.upper().str.strip()


def validate_deseq(df, gene_col, fc_col, padj_col, sample_col):
    required = [gene_col, fc_col, padj_col, sample_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing DESeq2 columns: {missing}")

    out = df.copy()
    out[gene_col] = normalize_gene_symbols(out[gene_col])
    out[fc_col] = pd.to_numeric(out[fc_col], errors="coerce")
    out[padj_col] = pd.to_numeric(out[padj_col], errors="coerce")
    out[sample_col] = out[sample_col].astype(str)
    return out


def validate_mapping(df, gene_col, set_col):
    required = [gene_col, set_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing mapping columns: {missing}")

    out = df.copy()
    out[gene_col] = normalize_gene_symbols(out[gene_col])
    out[set_col] = out[set_col].astype(str)
    out = out.dropna(subset=[gene_col, set_col]).drop_duplicates()
    return out


def build_sample_dict(df, sample_col="sample"):
    return {
        sample: sub.copy()
        for sample, sub in df.groupby(sample_col, sort=False)
    }


def fill_missing_genes(mapping_df, set_id, all_genes, current_df,
                       map_set_col="KE_ID_full", map_gene_col="gene_symbol",
                       current_gene_col="gene_symbol"):
    all_set_genes = mapping_df.loc[
        (mapping_df[map_set_col] == set_id) &
        (mapping_df[map_gene_col].isin(all_genes)),
        map_gene_col
    ]
    return sorted(set(all_set_genes) - set(current_df[current_gene_col]))


def prepare_signature_vector(
    df,
    genes,
    gene_col="gene_symbol",
    fc_col="log2FoldChange",
    padj_col="padj",
    weighted=True,
    epsilon=1e-300,
    min_weight=0,
    max_weight=10,
    padj_na_value=1.0
):
    d = df.copy()
    d[gene_col] = normalize_gene_symbols(d[gene_col])
    genes = [str(g).upper() for g in genes]
    d = d[d[gene_col].isin(genes)]

    if weighted:
        padj = d[padj_col].fillna(padj_na_value).clip(lower=epsilon)
        weights = (-np.log10(padj)).clip(lower=min_weight, upper=max_weight)
        d["score"] = d[fc_col] * weights
        value_col = "score"
    else:
        value_col = fc_col

    d = d.groupby(gene_col)[value_col].mean()

    vector = pd.Series(index=genes, dtype=float)
    vector.loc[d.index] = d.values
    return vector.fillna(0.0)


def compute_cosine_similarity_by_set(
    dict_samples,
    mapping_df,
    set_col="KE_ID_full",
    gene_col="gene_symbol",
    fc_col="log2FoldChange",
    padj_col="padj",
    weighted=True,
    min_genes=3,
    epsilon=1e-300,
    common_genes_only=True,
    min_weight=0,
    max_weight=10,
    return_diagnostics=False
):
    map_df = mapping_df.copy()
    map_df[gene_col] = normalize_gene_symbols(map_df[gene_col])

    sample_names = list(dict_samples.keys())
    matrices = {}
    diagnostics = []

    for set_id, sub_set in map_df.groupby(set_col):
        genes = sub_set[gene_col].dropna().unique().tolist()

        if common_genes_only:
            common = set(genes)
            for sample in sample_names:
                sample_genes = set(normalize_gene_symbols(dict_samples[sample][gene_col]).dropna())
                common &= sample_genes
            genes = sorted(common)

        if len(genes) < min_genes:
            continue

        vectors = []
        for sample in sample_names:
            vec = prepare_signature_vector(
                dict_samples[sample],
                genes=genes,
                gene_col=gene_col,
                fc_col=fc_col,
                padj_col=padj_col,
                weighted=weighted,
                epsilon=epsilon,
                min_weight=min_weight,
                max_weight=max_weight
            )
            vectors.append(vec)

        signature_matrix = pd.DataFrame(vectors, index=sample_names, columns=genes)

        vector_norms = np.linalg.norm(signature_matrix.values, axis=1)
        n_nonzero = (signature_matrix != 0).sum(axis=1)

        for sample, norm, nonzero in zip(sample_names, vector_norms, n_nonzero):
            diagnostics.append({
                "gene_set": set_id,
                "sample": sample,
                "n_genes_used": len(genes),
                "vector_norm": norm,
                "n_nonzero_genes": nonzero
            })

        # skip only if every vector is zero
        if (signature_matrix.abs().sum(axis=1) == 0).all():
            continue

        sim = cosine_similarity(signature_matrix)
        matrices[set_id] = pd.DataFrame(sim, index=sample_names, columns=sample_names)

    if return_diagnostics:
        return matrices, pd.DataFrame(diagnostics)
    return matrices


def circular_barplot_geneset(
    df,
    genes,
    title,
    sample_name,
    gene_set,
    gene_col="gene_symbol",
    fc_col="log2FoldChange",
    padj_col="padj",
    padj_thr=0.05,
    alphabetical=True,
    top_n=None,
    gap=0.03,
    min_genes_to_plot=3,
    scale_max=5,
    clip_max=3,
    label_fontsize=9,
    label_position=3.1,
    center_fontsize=10,
    center_box_frac=0.90,
    show_gene_labels=True,
    highlight_genes=None
):
    def gene_color(log2fc, padj):
        if pd.isna(padj) or padj > padj_thr:
            return "green"
        if log2fc > 0:
            return "#E69F00"
        if log2fc < 0:
            return "#0072B2"
        return "#D3D3D3"

    d = df.copy()
    d[gene_col] = normalize_gene_symbols(d[gene_col])
    genes_u = [str(g).upper() for g in genes]

    cols = [gene_col, fc_col] + ([padj_col] if padj_col in d.columns else [])
    sub = d[d[gene_col].isin(genes_u)][cols].dropna(subset=[fc_col])

    # add genes absent from sample as zero / nonsignificant
    present = set(sub[gene_col])
    missing = [g for g in genes_u if g not in present]
    if missing:
        sub = pd.concat([
            sub,
            pd.DataFrame({
                gene_col: missing,
                fc_col: 0.0,
                padj_col: 1.0
            })
        ], ignore_index=True)

    sub = sub.drop_duplicates(subset=[gene_col])

    if sub.shape[0] < min_genes_to_plot:
        raise ValueError(f"Only {sub.shape[0]} genes available for {gene_set}")

    if alphabetical:
        sub = sub.sort_values(gene_col)
    else:
        sub = sub.assign(_abs=sub[fc_col].abs()).sort_values("_abs", ascending=False).drop(columns="_abs")

    if top_n is not None:
        sub = sub.head(int(top_n))

    values = sub[fc_col].astype(float).values
    labels = sub[gene_col].tolist()
    padjs = sub[padj_col].values if padj_col in sub.columns else [None] * len(values)

    n = len(values)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = (2*np.pi/n) * (1-gap)
    values_clipped = np.clip(values, -clip_max, clip_max)

    r0 = scale_max * 1.2
    bottoms = np.where(values_clipped >= 0, r0, r0 + values_clipped)
    heights = np.abs(values_clipped)
    colors = [gene_color(fc, pj) for fc, pj in zip(values, padjs)]
    highlight_set = {str(g).upper() for g in (highlight_genes or [])}
    edgecolors = ["#7B1FA2" if str(g).upper() in highlight_set else "black" for g in labels]
    linewidths = [2.4 if str(g).upper() in highlight_set else 0.7 for g in labels]

    fig = plt.figure(figsize=(9.8, 9.8), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1], polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.bar(
        angles, heights, width=width, bottom=bottoms,
        color=colors, edgecolor=edgecolors, linewidth=linewidths,
        align="center", zorder=3
    )

    theta = np.linspace(0, 2*np.pi, 600)
    ax.plot(theta, np.full_like(theta, r0), color="black", linewidth=1.1, zorder=4)
    ax.set_xticks([])

    if show_gene_labels:
        label_radius = r0 + label_position
        for ang, label in zip(angles, labels):
            rotation = 90 - np.degrees(ang)
            ax.text(
                ang, label_radius, label,
                rotation=rotation, rotation_mode="anchor",
                ha="left", va="center",
                fontsize=label_fontsize, clip_on=False,
                fontweight="bold" if str(label).upper() in highlight_set else "normal",
                color="#7B1FA2" if str(label).upper() in highlight_set else "black"
            )

    tick_vals = np.array([-3, -1.5, 0, 1.5, 3])
    tick_vals = tick_vals[(tick_vals >= -scale_max) & (tick_vals <= scale_max)]
    ax.set_yticks(r0 + tick_vals)
    ax.set_yticklabels([""] * len(tick_vals))

    center_text = f"{sample_name}\n{gene_set}"
    r_minus3 = r0 - 3
    center_radius = max(0.1, r_minus3 * center_box_frac)

    ax.fill(theta, np.full_like(theta, center_radius), color="white", zorder=5)
    ax.text(
        0, 0, center_text, ha="center", va="center",
        fontsize=center_fontsize, fontweight="bold", zorder=6, wrap=True
    )

    ax.set_ylim(0, r0 + scale_max + 3)
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linewidth=0.5, alpha=0.45, zorder=1)
    ax.set_title(title, pad=30)
    return fig


def count_sig_up_down_geneset(
    df,
    genes,
    gene_col="gene_symbol",
    fc_col="log2FoldChange",
    padj_col="padj",
    padj_thr=0.05,
    fc_thr=0.0
):
    d = df.copy()
    d[gene_col] = normalize_gene_symbols(d[gene_col])
    genes_u = [str(g).upper() for g in genes]

    sub = d[d[gene_col].isin(genes_u)].copy()
    sub = sub.dropna(subset=[fc_col, padj_col]).drop_duplicates(subset=[gene_col])
    sig = sub[sub[padj_col] < padj_thr]
    up = sig[sig[fc_col] > fc_thr]
    down = sig[sig[fc_col] < -fc_thr]

    n_found = sub[gene_col].nunique()
    n_sig = sig[gene_col].nunique()

    return {
        "n_genes_in_geneset": len(set(genes_u)),
        "n_genes_found": n_found,
        "n_sig": n_sig,
        "n_up": up[gene_col].nunique(),
        "n_down": down[gene_col].nunique(),
        "pct_sig": (n_sig / n_found * 100) if n_found else 0,
        "up_genes": sorted(up[gene_col].unique().tolist()),
        "down_genes": sorted(down[gene_col].unique().tolist()),
        "sig_genes": sorted(sig[gene_col].unique().tolist())
    }


def deg_summary_for_samples(
    dict_samples,
    genes,
    gene_col="gene_symbol",
    fc_col="log2FoldChange",
    padj_col="padj",
    padj_thr=0.05,
    fc_thr=0.0
):
    rows = []
    for sample, df in dict_samples.items():
        res = count_sig_up_down_geneset(
            df, genes,
            gene_col=gene_col,
            fc_col=fc_col,
            padj_col=padj_col,
            padj_thr=padj_thr,
            fc_thr=fc_thr
        )
        rows.append({"sample": sample, **{k:v for k,v in res.items() if not isinstance(v, list)}})
    return pd.DataFrame(rows)


def plot_deg_grouped_bars(summary_df, padj_thr=0.05, title=None):
    d = summary_df.copy()
    x = np.arange(len(d))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7, len(d)*0.7), 4.5))
    ax.bar(
        x-width/2, d["n_up"], width=width,
        label=f"log2FC > 0 & padj ≤ {padj_thr}",
        color="#E69F00", edgecolor="black"
    )
    ax.bar(
        x+width/2, d["n_down"], width=width,
        label=f"log2FC < 0 & padj ≤ {padj_thr}",
        color="#0072B2", edgecolor="black"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(d["sample"], rotation=45, ha="right")
    ax.set_ylabel("Number of significant genes")
    if title:
        ax.set_title(title)
    ymax = max(d["n_up"].max(), d["n_down"].max(), 1)
    ax.set_ylim(0, ymax*1.35)

    for i, row in d.reset_index(drop=True).iterrows():
        ax.text(
            i, max(row["n_up"], row["n_down"]) + ymax*0.04,
            f'{row["pct_sig"]:.1f}%',
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_triangular_similarity_clean(
    sim_df,
    set_id=None,
    cmap="RdBu_r",
    figsize=(10, 10),
    method="average",
    fontsize_diag=7,
    fontsize_right=8
):
    """Clustered upper-triangular CSS heatmap with aligned dendrogram and labels."""
    sim_df = sim_df.copy()
    sim_df = sim_df.loc[sim_df.index, sim_df.index]

    dist = 1 - sim_df.values
    dist = np.clip(dist, 0, 2)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method, optimal_ordering=True)

    # Dedicated colorbar row prevents matplotlib from shrinking only the heatmap
    # axis, which previously displaced the right-hand sample labels.
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.35, 8, 0.42], width_ratios=[8, 2.2],
        hspace=0.03, wspace=0.02
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_labels = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
    ax_cbar = fig.add_subplot(gs[2, 0])

    dendro = dendrogram(
        Z, ax=ax_top, orientation="top", no_labels=True,
        color_threshold=0, above_threshold_color="#263238",
        link_color_func=lambda _: "#263238"
    )
    order = dendro["leaves"]
    sim_ordered = sim_df.iloc[order, order]
    labels = sim_ordered.index.tolist()
    n = len(labels)

    x_max = 10 * n
    ax_top.set_xlim(0, x_max)
    ax_top.margins(x=0)
    ax_top.axis("off")

    data = sim_ordered.values
    mask = np.tril(np.ones_like(data, dtype=bool), k=0)
    data_masked = np.ma.array(data, mask=mask)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    im = ax_heatmap.imshow(
        data_masked, cmap=cmap_obj, vmin=-1, vmax=1,
        origin="upper", extent=[0, x_max, x_max, 0],
        aspect="auto", interpolation="nearest"
    )
    ax_heatmap.set_xlim(0, x_max)
    ax_heatmap.set_ylim(x_max, 0)
    ax_heatmap.margins(x=0, y=0)

    for i in range(n + 1):
        pos = i * 10
        ax_heatmap.axhline(pos, color="white", linewidth=0.55)
        ax_heatmap.axvline(pos, color="white", linewidth=0.55)

    for i, label in enumerate(labels):
        ax_heatmap.text(
            i * 10 + 5, i * 10 + 9.2, label,
            ha="center", va="top", rotation=90,
            fontsize=fontsize_diag, clip_on=True
        )

    ax_heatmap.axis("off")
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(x_max, 0)
    ax_labels.margins(y=0)
    ax_labels.axis("off")
    for i, label in enumerate(labels):
        ax_labels.text(
            0.02, i * 10 + 5, label,
            ha="left", va="center", fontsize=fontsize_right, clip_on=False
        )

    fig.suptitle(
        f"Clustered cosine similarity — {set_id}" if set_id else "Clustered cosine similarity",
        y=0.985, fontsize=13, fontweight="semibold"
    )

    cb = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cb.set_label("Cosine similarity", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.07)

    return sim_ordered, Z, fig


def extract_chemical(name, separator="_", chemical_level=1):
    parts = str(name).split(separator)
    return separator.join(parts[:chemical_level])


def median_css_all_pairs(
    sim_df,
    separator="_",
    chemical_level=1,
    exclude_same_chemical=True
):
    rows = []
    samples = sim_df.index.tolist()

    for i, s1 in enumerate(samples):
        c1 = extract_chemical(s1, separator, chemical_level)
        for j in range(i+1, len(samples)):
            s2 = samples[j]
            c2 = extract_chemical(s2, separator, chemical_level)

            if exclude_same_chemical and c1 == c2:
                continue

            rows.append({
                "sample_1": s1,
                "sample_2": s2,
                "chemical_1": c1,
                "chemical_2": c2,
                "cosine_similarity": sim_df.iloc[i, j]
            })

    raw = pd.DataFrame(rows).dropna(subset=["cosine_similarity"])
    if raw.empty:
        return raw

    raw["pair"] = raw.apply(
        lambda x: tuple(sorted([x["chemical_1"], x["chemical_2"]])),
        axis=1
    )

    summary = (
        raw.groupby("pair")["cosine_similarity"]
        .agg(
            median_cosine_similarity="median",
            mean_cosine_similarity="mean",
            min_cosine_similarity="min",
            max_cosine_similarity="max",
            n_comparisons="size"
        )
        .reset_index()
    )

    summary[["chemical_1", "chemical_2"]] = pd.DataFrame(
        summary["pair"].tolist(), index=summary.index
    )

    return (
        summary[
            [
                "chemical_1", "chemical_2",
                "median_cosine_similarity",
                "mean_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
                "n_comparisons"
            ]
        ]
        .sort_values("median_cosine_similarity", ascending=False)
        .reset_index(drop=True)
    )


def silhouette_from_similarity(
    sim_df,
    group_definitions,
    separator="_",
    chemical_level=1
):
    chem_to_group = {}
    group_names = list(group_definitions.keys())

    for gi, (group_name, chemicals) in enumerate(group_definitions.items()):
        for chem in chemicals:
            chem_to_group[chem] = gi

    keep = []
    labels = []

    for sample in sim_df.index:
        chem = extract_chemical(sample, separator, chemical_level)
        if chem in chem_to_group:
            keep.append(sample)
            labels.append(chem_to_group[chem])

    if len(set(labels)) < 2:
        raise ValueError("At least two represented groups are required.")
    if len(keep) <= len(set(labels)):
        raise ValueError("Not enough samples to compute a silhouette score.")

    selected = sim_df.loc[keep, keep].copy().clip(-1, 1)
    dist = 1 - selected
    np.fill_diagonal(dist.values, 0)

    overall = silhouette_score(dist, labels, metric="precomputed")
    sample_scores = silhouette_samples(dist, labels, metric="precomputed")

    sample_df = pd.DataFrame({
        "sample": keep,
        "chemical": [extract_chemical(s, separator, chemical_level) for s in keep],
        "group": [group_names[i] for i in labels],
        "silhouette_score": sample_scores
    })

    chem_df = (
        sample_df.groupby(["group", "chemical"])["silhouette_score"]
        .agg(["median", "mean", "min", "max", "count"])
        .reset_index()
        .sort_values(["group", "median"], ascending=[True, False])
    )

    return overall, sample_df, chem_df
