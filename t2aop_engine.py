
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    max_weight=5,
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
    max_weight=5,
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


_LABEL_STRIP_RE = re.compile(
    r"_(DMSO|MED|MEDIUM|NaCl|NACL|CTRL|CONTROL)[ _]?[A-Za-z]?\d*$",
    flags=re.IGNORECASE
)


def clean_sample_label(name, overrides=None):
    """Turn a raw sample/file identifier into a publication-ready label.

    Strips a trailing .csv, removes the solvent/control suffix
    (e.g. '_DMSO_C0', '_MED C0', '_NaClR'), and applies any user-supplied
    overrides. `overrides` is a {raw_or_cleaned_name: wanted_label} mapping
    and always wins over the automatic cleaning.
    """
    raw = str(name)
    overrides = overrides or {}
    if raw in overrides and str(overrides[raw]).strip():
        return str(overrides[raw]).strip()

    s = re.sub(r"\.csv$", "", raw, flags=re.IGNORECASE)
    s = _LABEL_STRIP_RE.sub("", s)
    s = s.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)

    if s in overrides and str(overrides[s]).strip():
        return str(overrides[s]).strip()
    return s


def circular_barplot_geneset(
    df,
    genes,
    title,
    sample_name,
    gene_set,
    ax=None,
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
    show_center_text=True,
    highlight_genes=None,
    up_color="#E69F00",
    down_color="#0072B2",
    nonsig_color="green",
    missing_color="#8A9099"
):
    def gene_color(log2fc, padj):
        if pd.isna(padj) or padj > padj_thr:
            return nonsig_color
        if log2fc > 0:
            return up_color
        if log2fc < 0:
            return down_color
        return nonsig_color

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
    missing_set = set(missing)
    colors = [missing_color if g in missing_set else gene_color(fc, pj)
              for g, fc, pj in zip(labels, values, padjs)]
    highlight_set = {str(g).upper() for g in (highlight_genes or [])}
    edgecolors = ["#7B1FA2" if str(g).upper() in highlight_set else "black" for g in labels]
    linewidths = [2.4 if str(g).upper() in highlight_set else 0.7 for g in labels]

    if ax is None:
        fig = plt.figure(figsize=(9.8, 9.8), dpi=160)
        ax = fig.add_axes([0, 0, 1, 1], polar=True)
    else:
        fig = ax.figure
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
                color=("#A0A6B0" if label in missing_set else
                       "#7B1FA2" if str(label).upper() in highlight_set else "black")
            )

    tick_vals = np.array([-3, -1.5, 0, 1.5, 3])
    tick_vals = tick_vals[(tick_vals >= -scale_max) & (tick_vals <= scale_max)]
    ax.set_yticks(r0 + tick_vals)
    ax.set_yticklabels([""] * len(tick_vals))

    center_text = f"{sample_name}\n{gene_set}"
    r_minus3 = r0 - 3
    center_radius = max(0.1, r_minus3 * center_box_frac)

    ax.fill(theta, np.full_like(theta, center_radius), color="white", zorder=5)
    if show_center_text:
        ax.text(
            0, 0, center_text, ha="center", va="center",
            fontsize=center_fontsize, fontweight="bold", zorder=6, wrap=True
        )

    # Without gene labels there is no need to reserve the outer label ring:
    # tightening the radial limit makes the circle fill its axes, which
    # matters a lot in multi-panel figures.
    if show_gene_labels:
        ax.set_ylim(0, r0 + scale_max + 3)
    else:
        ax.set_ylim(0, r0 + min(scale_max, clip_max) + 0.4)
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linewidth=0.5, alpha=0.45, zorder=1)
    ax.set_title(title, pad=30)
    return fig


def circular_panel_grid(
    sample_dict,
    genes_by_set,
    samples,
    gene_sets,
    label_overrides=None,
    gene_col="gene_symbol",
    fc_col="log2FoldChange",
    padj_col="padj",
    padj_thr=0.05,
    alphabetical=True,
    clip_max=3,
    scale_max=5,
    show_gene_labels=False,
    highlight_genes=None,
    panel_size=3.4,
    dpi=200,
    row_label_fontsize=13,
    col_label_fontsize=13,
    show_legend=True,
    up_color="#E69F00",
    down_color="#0072B2",
    nonsig_color="green",
    missing_color="#8A9099"
):
    """Assemble circular T2AOP profiles into one publication-ready panel.

    Rows are gene sets, columns are samples. Every circle is drawn on the
    same radial scale, so bar lengths are directly comparable across the
    whole figure. `genes_by_set` maps each gene set id to the list of genes
    to plot for that row, so the same gene occupies the same angular
    position in every panel of that row.
    """
    samples = list(samples)
    gene_sets = list(gene_sets)
    if not samples or not gene_sets:
        raise ValueError("Select at least one sample and one gene set.")

    n_rows, n_cols = len(gene_sets), len(samples)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows + (0.6 if show_legend else 0)),
        dpi=dpi,
        subplot_kw={"polar": True},
        squeeze=False
    )

    for i, set_id in enumerate(gene_sets):
        genes = list(genes_by_set.get(set_id, []))
        for j, sample in enumerate(samples):
            ax = axes[i][j]
            try:
                circular_barplot_geneset(
                    sample_dict[sample],
                    genes=genes,
                    title="",
                    sample_name=sample,
                    gene_set=set_id,
                    ax=ax,
                    gene_col=gene_col,
                    fc_col=fc_col,
                    padj_col=padj_col,
                    padj_thr=padj_thr,
                    alphabetical=alphabetical,
                    clip_max=clip_max,
                    scale_max=scale_max,
                    show_gene_labels=show_gene_labels,
                    show_center_text=False,
                    highlight_genes=highlight_genes,
                    up_color=up_color,
                    down_color=down_color,
                    nonsig_color=nonsig_color,
                    missing_color=missing_color
                )
            except Exception as exc:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0, 0, f"n/a\n{exc}", ha="center", va="center",
                    fontsize=7, color="#99999E"
                )

            if i == 0:
                ax.set_title(
                    clean_sample_label(sample, label_overrides),
                    fontsize=col_label_fontsize, fontweight="bold", pad=18
                )
            if j == 0:
                ax.text(
                    -0.12, 0.5, set_id,
                    transform=ax.transAxes, rotation=90,
                    ha="center", va="center",
                    fontsize=row_label_fontsize, fontweight="bold"
                )

    if show_legend:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=up_color, edgecolor="black", linewidth=0.7),
            plt.Rectangle((0, 0), 1, 1, facecolor=down_color, edgecolor="black", linewidth=0.7),
            plt.Rectangle((0, 0), 1, 1, facecolor=nonsig_color, edgecolor="black", linewidth=0.7),
            plt.Rectangle((0, 0), 1, 1, facecolor=missing_color, edgecolor="black", linewidth=0.7),
        ]
        labels = [
            f"Upregulated (padj ≤ {padj_thr})",
            f"Downregulated (padj ≤ {padj_thr})",
            "Not significant",
            "Not detected in sample",
        ]
        fig.legend(
            handles, labels,
            loc="lower center", ncol=4, frameon=False,
            fontsize=10, bbox_to_anchor=(0.5, -0.005)
        )

    top = 1 - 0.5 / (panel_size * n_rows)
    bottom = (0.55 if show_legend else 0.08) / (panel_size * n_rows)
    fig.subplots_adjust(
        left=0.04, right=0.995, top=top, bottom=bottom,
        wspace=0.02, hspace=0.06 if show_gene_labels else 0.02
    )
    return fig



def _prepare_profile(df, genes, gene_set, gene_col="gene_symbol",
                     fc_col="log2FoldChange", padj_col="padj",
                     alphabetical=True, top_n=None, min_genes_to_plot=3,
                     gene_order=None):
    """Shared data preparation for the circular and flat profiles."""
    d = df.copy()
    d[gene_col] = normalize_gene_symbols(d[gene_col])
    genes_u = [str(g).upper() for g in genes]

    cols = [gene_col, fc_col] + ([padj_col] if padj_col in d.columns else [])
    sub = d[d[gene_col].isin(genes_u)][cols].dropna(subset=[fc_col])

    present = set(sub[gene_col])
    missing = [g for g in genes_u if g not in present]
    if missing:
        sub = pd.concat([sub, pd.DataFrame({gene_col: missing, fc_col: 0.0,
                                            padj_col: 1.0})], ignore_index=True)
    sub = sub.drop_duplicates(subset=[gene_col])
    if sub.shape[0] < min_genes_to_plot:
        raise ValueError(f"Only {sub.shape[0]} genes available for {gene_set}")

    if gene_order is not None:
        # an explicit order was supplied: honour it exactly, so that paired
        # panels keep every gene at the same position
        order = [str(g).upper() for g in gene_order]
        sub = (sub.set_index(gene_col).reindex(order).dropna(subset=[fc_col])
                  .rename_axis(gene_col).reset_index())
    elif alphabetical:
        sub = sub.sort_values(gene_col)
    else:
        sub = (sub.assign(_abs=sub[fc_col].abs())
                  .sort_values("_abs", ascending=False).drop(columns="_abs"))
    if top_n is not None:
        sub = sub.head(int(top_n))

    labels = sub[gene_col].tolist()
    values = sub[fc_col].astype(float).values
    padjs = sub[padj_col].values if padj_col in sub.columns else [None] * len(values)
    return labels, values, padjs, set(missing)


def flat_barplot_geneset(
    df, genes, title, sample_name, gene_set,
    ax=None,
    gene_col="gene_symbol", fc_col="log2FoldChange", padj_col="padj",
    padj_thr=0.05, alphabetical=True, top_n=None, gap=0.03,
    min_genes_to_plot=3, scale_max=5, clip_max=3,
    label_fontsize=7, label_position=None, center_fontsize=10,
    center_box_frac=None, show_gene_labels=True, show_center_text=True,
    highlight_genes=None,
    up_color="#E69F00", down_color="#0072B2",
    nonsig_color="green", missing_color="#8A9099",
    mirror=False, gene_order=None, label_only_highlight=False
):
    """Flat (aligned) version of the circular profile.

    Genes keep a fixed position on the x axis, so the same gene sits at the
    same place in every panel and profiles can be compared column by column.
    `mirror=True` draws the bars downward, for paired comparisons.
    """
    labels, values, padjs, missing_set = _prepare_profile(
        df, genes, gene_set, gene_col, fc_col, padj_col,
        alphabetical, top_n, min_genes_to_plot, gene_order=gene_order)

    def gene_color(fc, pj):
        if pd.isna(pj) or pj > padj_thr:
            return nonsig_color
        return up_color if fc > 0 else down_color if fc < 0 else nonsig_color

    colors = [missing_color if g in missing_set else gene_color(fc, pj)
              for g, fc, pj in zip(labels, values, padjs)]
    hl = {str(g).upper() for g in (highlight_genes or [])}
    if label_only_highlight:
        edges = ["black"] * len(labels)
        widths = [0.35] * len(labels)
    else:
        edges = ["#7B1FA2" if str(g).upper() in hl else "black" for g in labels]
        widths = [1.3 if str(g).upper() in hl else 0.35 for g in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.11), 3.2), dpi=160)
    else:
        fig = ax.figure

    v = np.clip(values, -clip_max, clip_max)
    x = np.arange(len(labels))
    ax.bar(x, v, width=1 - gap, color=colors, edgecolor=edges,
           linewidth=widths, zorder=3)
    ax.axhline(0, color="black", linewidth=1.0, zorder=4)

    lim = min(scale_max, clip_max) * 1.08
    ax.set_ylim((lim, -lim) if mirror else (-lim, lim))
    ax.set_xlim(-0.8, len(labels) - 0.2)
    ax.set_ylabel("log2FC", fontsize=9)

    if show_gene_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
        for tick, g in zip(ax.get_xticklabels(), labels):
            if g in missing_set:
                tick.set_color("#A0A6B0")
            if str(g).upper() in hl:
                tick.set_color("#7B1FA2"); tick.set_fontweight("bold")
    else:
        ax.set_xticks([])

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    if show_center_text:
        ax.set_title(f"{sample_name} — {gene_set}", fontsize=center_fontsize,
                     fontweight="bold", pad=6)
    elif title:
        ax.set_title(title, fontsize=center_fontsize, pad=6)
    return fig


def profile_panel_grid(
    sample_dict, genes_by_set, samples, gene_sets,
    style="circular", label_overrides=None, **kwargs
):
    """Panel of profiles, drawn either as circles or as flat aligned bars."""
    if style == "flat":
        return _flat_panel_grid(sample_dict, genes_by_set, samples, gene_sets,
                                label_overrides=label_overrides, **kwargs)
    return circular_panel_grid(sample_dict, genes_by_set, samples, gene_sets,
                               label_overrides=label_overrides, **kwargs)


def _flat_panel_grid(
    sample_dict, genes_by_set, samples, gene_sets,
    label_overrides=None, gene_col="gene_symbol", fc_col="log2FoldChange",
    padj_col="padj", padj_thr=0.05, alphabetical=True, clip_max=3, scale_max=5,
    show_gene_labels=False, highlight_genes=None, panel_size=3.4, dpi=200,
    row_label_fontsize=13, col_label_fontsize=13, show_legend=True,
    up_color="#E69F00", down_color="#0072B2",
    nonsig_color="green", missing_color="#8A9099"
):
    samples, gene_sets = list(samples), list(gene_sets)
    if not samples or not gene_sets:
        raise ValueError("Select at least one sample and one gene set.")

    n_rows, n_cols = len(gene_sets), len(samples)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * 1.5 * n_cols, panel_size * 0.78 * n_rows +
                 (0.6 if show_legend else 0)),
        dpi=dpi, squeeze=False, sharey="row",
        gridspec_kw={"hspace": 0.55 if show_gene_labels else 0.32, "wspace": 0.12})

    for i, set_id in enumerate(gene_sets):
        genes = sorted(genes_by_set.get(set_id, []))   # one order for the row
        for j, sample in enumerate(samples):
            ax = axes[i][j]
            try:
                flat_barplot_geneset(
                    sample_dict[sample], genes=genes, title="",
                    sample_name=sample, gene_set=set_id, ax=ax,
                    gene_col=gene_col, fc_col=fc_col, padj_col=padj_col,
                    padj_thr=padj_thr, alphabetical=True,
                    gene_order=genes,
                    clip_max=clip_max, scale_max=scale_max,
                    show_gene_labels=show_gene_labels, show_center_text=False,
                    highlight_genes=highlight_genes, up_color=up_color,
                    down_color=down_color, nonsig_color=nonsig_color,
                    missing_color=missing_color)
            except Exception as exc:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, f"n/a\n{exc}", ha="center", va="center",
                        fontsize=7, color="#99999E", transform=ax.transAxes)
            if j != 0:
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(clean_sample_label(sample, label_overrides),
                             fontsize=col_label_fontsize, fontweight="bold", pad=8)
            if j == 0:
                ax.text(-0.17, 0.5, set_id, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=row_label_fontsize,
                        fontweight="bold")

    if show_legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                                 linewidth=0.7)
                   for c in (up_color, down_color, nonsig_color, missing_color)]
        labels = [f"Upregulated (padj \u2264 {padj_thr})",
                  f"Downregulated (padj \u2264 {padj_thr})",
                  "Not significant", "Not detected in sample"]
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
    return fig


def dual_flat_comparison(
    df_a, df_b, genes, name_a, name_b, gene_set,
    gene_col="gene_symbol", fc_col="log2FoldChange", padj_col="padj",
    padj_thr=0.05, clip_max=3, scale_max=5, show_gene_labels=False,
    highlight_genes=None, order_by="magnitude", dpi=200,
    up_color="#E69F00", down_color="#0072B2",
    nonsig_color="green", missing_color="#8A9099",
    mark_concordance=True,
    concordant_band="#DCD3EC", discordant_band="#F3D9D2"
):
    """Two samples as mirrored flat profiles, gene positions aligned.

    With mark_concordance, every gene significant in both samples is marked
    behind the bars: a band in both panels, plus a hatch on the bars of
    genes that move in opposite directions.
    """
    la, va, pa, ma = _prepare_profile(df_a, genes, gene_set, gene_col, fc_col,
                                      padj_col, alphabetical=True)
    lb, vb, pb, mb = _prepare_profile(df_b, genes, gene_set, gene_col, fc_col,
                                      padj_col, alphabetical=True)
    A = pd.DataFrame({"g": la, "fc": va, "pj": pa})
    B = pd.DataFrame({"g": lb, "fc": vb, "pj": pb})
    m = A.merge(B, on="g", suffixes=("_a", "_b"))
    if order_by == "alphabetical":
        m = m.sort_values("g")
    elif order_by == "concordance":
        m = m.assign(_c=m["fc_a"] * m["fc_b"]).sort_values("_c", ascending=False)
        m = m.drop(columns="_c")
    else:                                  # "magnitude": by the first sample
        m = m.sort_values("fc_a", ascending=False)
    m = m.reset_index(drop=True)
    shared_order = m["g"].tolist()

    shared = ((pd.to_numeric(m["pj_a"], errors="coerce") <= padj_thr) &
              (pd.to_numeric(m["pj_b"], errors="coerce") <= padj_thr) &
              (np.sign(m["fc_a"]) == np.sign(m["fc_b"])) & (m["fc_a"] != 0))
    # direction agreement, independent of significance
    has_dir = (m["fc_a"] != 0) & (m["fc_b"] != 0)
    same_dir = has_dir & (np.sign(m["fc_a"]) == np.sign(m["fc_b"]))
    opposed = has_dir & (np.sign(m["fc_a"]) != np.sign(m["fc_b"]))
    both_sig = (has_dir &
                (pd.to_numeric(m["pj_a"], errors="coerce") <= padj_thr) &
                (pd.to_numeric(m["pj_b"], errors="coerce") <= padj_thr))
    sig_opposed = opposed & both_sig
    hl = set(m.loc[shared, "g"]) if highlight_genes is None else {
        str(g).upper() for g in highlight_genes}

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(max(7.0, len(m) * 0.11), 4.4), dpi=dpi, sharex=True,
        gridspec_kw={"hspace": 0.08 if not show_gene_labels else 0.45})

    for ax, suf, name, mirror in ((axT, "_a", name_a, False),
                                  (axB, "_b", name_b, True)):
        sub = pd.DataFrame({gene_col: m["g"], fc_col: m["fc" + suf],
                            padj_col: m["pj" + suf]})
        flat_barplot_geneset(
            sub, genes=m["g"].tolist(), title="", sample_name=name,
            gene_set=gene_set, ax=ax, gene_col=gene_col, fc_col=fc_col,
            padj_col=padj_col, padj_thr=padj_thr, alphabetical=False,
            gene_order=shared_order,
            clip_max=clip_max, scale_max=scale_max,
            show_gene_labels=show_gene_labels and mirror,
            show_center_text=False, highlight_genes=hl, mirror=mirror,
            label_only_highlight=mark_concordance,
            up_color=up_color, down_color=down_color,
            nonsig_color=nonsig_color, missing_color=missing_color)
        ax.set_ylabel(f"{name}\nlog2FC", fontsize=9, fontweight="bold")

        if mark_concordance:
            for i in range(len(m)):
                if not has_dir.iloc[i]:
                    continue
                ax.axvspan(i - 0.5, i + 0.5, zorder=0, linewidth=0,
                           color=discordant_band if opposed.iloc[i]
                           else concordant_band)
            bars = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)
                    and p.get_zorder() >= 2]
            if len(bars) == len(m):
                for i, bar in enumerate(bars):
                    if sig_opposed.iloc[i]:
                        bar.set_hatch("////")
                        bar.set_edgecolor("#5A3A32")
                        bar.set_linewidth(0.7)

    if mark_concordance:
        handles = [
            mpatches.Patch(facecolor=concordant_band, edgecolor="none",
                           label="same direction"),
            mpatches.Patch(facecolor=discordant_band, edgecolor="none",
                           label="opposite directions"),
            mpatches.Patch(facecolor="white", edgecolor="#5A3A32",
                           hatch="////", linewidth=0.7,
                           label="significant in both, opposed"),
            mpatches.Patch(facecolor="none", edgecolor="none",
                           label="bold purple symbol = significant in both, same direction"),
        ]
        axT.legend(handles=handles, frameon=False, fontsize=7.5, ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, 1.00),
                   handlelength=1.4, columnspacing=1.4)

    axT.set_title(
        f"{name_a} vs {name_b}  —  {gene_set}   |   {len(m)} shared genes;  "
        f"{int(same_dir.sum())} same direction "
        f"({int(shared.sum())} significant in both), "
        f"{int(opposed.sum())} opposite ({int(sig_opposed.sum())} significant in both)",
        fontsize=10, fontweight="bold", pad=24 if mark_concordance else 8)
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
    s = str(name)
    if separator and separator in s:
        parts = s.split(separator)
        return separator.join(parts[:chemical_level])
    # Cleaned display labels use spaces ("MMS C5"), so fall back to
    # whitespace/underscore splitting when the chosen separator is absent.
    parts = [p for p in re.split(r"[\s_]+", s) if p]
    return " ".join(parts[:chemical_level]) if parts else s


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


def exact_silhouette_permutations(sim_df, group_definitions, separator="_", chemical_level=1):
    """Enumerate all distinct compound assignments preserving group sizes.

    Scores are computed on the existing sample-level distance matrix, with
    all samples of a compound assigned together. Includes observed assignment.
    """
    from itertools import combinations
    from math import factorial
    chemical_group = {}
    group_names = list(group_definitions)
    for gi, (group, compounds) in enumerate(group_definitions.items()):
        for compound in compounds:
            if compound in chemical_group:
                raise ValueError(f"Compound '{compound}' appears in multiple groups.")
            chemical_group[compound] = gi
    samples = [s for s in sim_df.index if extract_chemical(s, separator, chemical_level) in chemical_group]
    chemicals = sorted(set(extract_chemical(s, separator, chemical_level) for s in samples))
    labels = tuple(chemical_group[c] for c in chemicals)
    sizes = [labels.count(i) for i in range(len(group_names))]
    if len(chemicals) < 3 or any(x == 0 for x in sizes) or len(sizes) < 2:
        raise ValueError("At least two nonempty groups and three compounds are required.")
    total = factorial(len(chemicals))
    for size in sizes:
        total //= factorial(size)
    if total > 20000:
        raise ValueError(f"{total:,} assignments; exact enumeration is limited to 20,000.")
    dist = 1 - sim_df.loc[samples, samples].to_numpy(dtype=float).clip(-1, 1)
    np.fill_diagonal(dist, 0)
    if not np.isfinite(dist).all():
        raise ValueError("Similarity matrix contains missing or non-finite values.")
    indices = np.array([chemicals.index(extract_chemical(s, separator, chemical_level)) for s in samples])
    rows = []
    def assignments(gi, remaining, current):
        if gi == len(sizes) - 1:
            arr = current.copy()
            for k in remaining:
                arr[k] = gi
            yield tuple(arr)
        else:
            for selected in combinations(remaining, sizes[gi]):
                arr = current.copy()
                for k in selected:
                    arr[k] = gi
                remaining_next = [k for k in remaining if k not in selected]
                yield from assignments(gi + 1, remaining_next, arr)
    for assignment in assignments(0, list(range(len(chemicals))), [-1] * len(chemicals)):
        sample_labels = np.asarray(assignment)[indices]
        score = float(silhouette_score(dist, sample_labels, metric="precomputed"))
        row = {"silhouette_score": score, "observed_assignment": assignment == labels}
        for gi, name in enumerate(group_names):
            row[f"group_{name}"] = ", ".join(c for c, lab in zip(chemicals, assignment) if lab == gi)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("silhouette_score", ascending=False).reset_index(drop=True)
    observed = float(df.loc[df.observed_assignment, "silhouette_score"].iloc[0])
    extreme = int((df.silhouette_score >= observed - 1e-12).sum())
    return {"observed": observed, "p_value": extreme / len(df), "n_extreme": extreme,
            "n_assignments": len(df), "n_chemicals": len(chemicals),
            "n_samples": len(samples), "null_mean": float(df.silhouette_score.mean()),
            "distribution": df}

def permutation_test_silhouette(sim_df, group_definitions, separator="_", chemical_level=1,
                                n_permutations=999, random_state=42):
    """Test whether predefined chemical groups exceed random chemical-level assignments.

    Permutes whole-chemical labels while preserving numbers of chemicals per group.
    The cosine distance matrix and all sample/concentration replicates stay fixed.
    """
    from collections import Counter
    from math import factorial

    chemical_group = {}
    group_names = list(group_definitions)
    for gi, (group, chemicals) in enumerate(group_definitions.items()):
        for chemical in chemicals:
            if chemical in chemical_group:
                raise ValueError(f"Chemical '{chemical}' belongs to multiple groups.")
            chemical_group[chemical] = gi

    samples = [s for s in sim_df.index if extract_chemical(s, separator, chemical_level) in chemical_group]
    chems_per_sample = [extract_chemical(s, separator, chemical_level) for s in samples]
    chemicals = sorted(set(chems_per_sample))
    if len(chemicals) < 3:
        raise ValueError("At least three represented chemicals are required for a permutation test.")
    chemical_labels = np.array([chemical_group[c] for c in chemicals], dtype=int)
    counts = Counter(chemical_labels)
    if len(counts) < 2:
        raise ValueError("At least two represented groups are required.")
    if len(samples) <= len(counts):
        raise ValueError("Not enough samples for silhouette analysis.")
    dist = 1 - sim_df.loc[samples, samples].to_numpy(dtype=float).clip(-1, 1)
    np.fill_diagonal(dist, 0)
    if not np.isfinite(dist).all():
        raise ValueError("Similarity matrix contains missing or non-finite values.")
    chem_idx = np.array([chemicals.index(c) for c in chems_per_sample], dtype=int)

    def score(labels):
        sample_labels = labels[chem_idx]
        if len(np.unique(sample_labels)) >= len(samples):
            return np.nan
        return float(silhouette_score(dist, sample_labels, metric="precomputed"))

    observed = score(chemical_labels)
    total = factorial(len(chemicals))
    for n in counts.values():
        total //= factorial(n)
    if total <= 1:
        raise ValueError("No alternative chemical-group assignments are possible.")
    rng = np.random.default_rng(int(random_state))
    target = min(int(n_permutations), total - 1)
    if target < 1:
        raise ValueError("Request at least one permutation.")
    original = tuple(chemical_labels.tolist())
    null = []
    seen = {original}
    # Exhaustive enumeration for small group assignments; otherwise sample distinct assignments.
    if total <= 20000:
        from itertools import combinations
        def assignments(group_i, remaining, prefix):
            if group_i == len(counts) - 1:
                out = prefix.copy()
                for idx in remaining:
                    out[idx] = group_i
                yield tuple(out)
                return
            for chosen in combinations(remaining, counts[group_i]):
                out = prefix.copy()
                for idx in chosen:
                    out[idx] = group_i
                rest = [idx for idx in remaining if idx not in chosen]
                yield from assignments(group_i + 1, rest, out)
        possibilities = [a for a in assignments(0, list(range(len(chemicals))), [-1]*len(chemicals)) if a != original]
        indices = rng.choice(len(possibilities), size=target, replace=False)
        for i in indices:
            null.append(score(np.asarray(possibilities[int(i)], dtype=int)))
    else:
        attempts = 0
        while len(null) < target and attempts < target * 30 + 1000:
            attempts += 1
            candidate = tuple(rng.permutation(chemical_labels).tolist())
            if candidate in seen:
                continue
            seen.add(candidate)
            null.append(score(np.asarray(candidate, dtype=int)))
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null):
        raise ValueError("No valid permutations were generated.")
    p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (1 + len(null))
    return {
        "observed": observed, "p_value": float(p), "null_scores": null,
        "n_permutations": len(null), "total_assignments": total,
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)) if len(null)>1 else 0.0,
        "exact": len(null) == total - 1,
        "n_chemicals": len(chemicals), "n_samples": len(samples)
    }
