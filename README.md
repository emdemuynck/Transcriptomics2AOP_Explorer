# Transcriptomics2AOP Explorer v3

This version was rebuilt around the analysis workflow in
`transcriptomics2AOP_GENOTOX_analysis.ipynb`.

## Current modules

- Combined DESeq2 input with a `sample` column
- Flexible column mapping
- T2AOP gene-set / KE / branch selection
- Circular gene-expression profiles
- Significant up/down gene counts
- Weighted cosine similarity
  - score = `log2FC × -log10(padj)`
  - default maximum weight = 10
  - optional common-gene restriction
- Triangular CSS heatmap + hierarchical clustering
- Median CSS between chemicals
- User-defined mechanistic groups
- Overall silhouette score
- Silhouette scores by sample and chemical
- Similarity-vector diagnostics
- CSV and PNG export

## DESeq2 input

One combined CSV / TSV / XLSX file. Minimum required information:

- `gene_symbol`
- `log2FoldChange`
- `padj`
- `sample`

Exact names are not mandatory because the app lets you map the columns.

## Mapping input

Minimum:

- gene symbol
- T2AOP gene-set / KE / branch identifier

Examples of identifiers can include:

- `Clastogenicity`
- `Aneugenicity`
- `Stress pathway`
- `GENOMARK`
- `TGx-DDI`
- individual KEs

The app does not hard-code these names: whatever is present in the mapping file
becomes selectable.

## Install / run on Windows

From PowerShell in this folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL displayed by Streamlit, usually:

`http://localhost:8501`

## Methodological defaults taken from the notebook

- DEGs: `padj < 0.05`
- CSS weighting: enabled
- weight: `-log10(padj)`
- maximum weight: `10`
- missing values in signature vectors: `0`
- cosine-similarity range: `-1` to `1`
- hierarchical linkage: `average`

All important parameters are exposed in the interface.


## v4.2 circular-profile update

Circular profiles now use only the intersection of gene-set genes present in all
samples selected in the Circular profiles tab. This guarantees that every
displayed sample is plotted using the exact same gene universe. The interface
reports the number of common genes retained relative to the full gene set.
