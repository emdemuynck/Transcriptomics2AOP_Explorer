# Transcriptomics2AOP Explorer

Interactive application accompanying:

> Demuynck E., Thienpont A., Vanhaecke T., Mertens B. *Mapping transcriptomic
> biomarkers on an adverse outcome pathway network for genotoxicity.* ALTEX (submitted).

The Explorer organises differential gene expression results according to gene sets
anchored to key events of an adverse outcome pathway (AOP) network, and implements the
three analytical components of the Transcriptomics2AOP (T2AOP) workflow: **profiling**,
**comparing** and **clustering**.

It is an exploratory analysis and visualisation tool. It does not establish that a key
event has occurred, does not identify mechanisms of action, and is not intended to
support regulatory classification.

## Running the application

```bash
git clone https://github.com/emdemuynck/Transcriptomics2AOP.git
cd Transcriptomics2AOP
pip install -r requirements.txt
streamlit run app.py
```

Python 3.9 or later.

On first launch the sidebar offers **Load example data**, which loads the small synthetic
dataset in `examples/` so that the interface can be explored without uploading anything.
The example carries no biological meaning.

## Input files

Two plain-text files are required. Column names are mapped in the interface, so the
exact headers do not matter.

**1. Differential expression table** — one combined CSV, TSV or XLSX file containing,
for every sample, at least:

| column | content |
|---|---|
| `sample` | sample identifier, e.g. `AFB1_C4` |
| `gene_symbol` | HGNC gene symbol |
| `log2FoldChange` | log2 fold change versus the matched control |
| `padj` | adjusted p value |

**2. Gene-set mapping** — a two-column file assigning genes to gene sets:

| column | content |
|---|---|
| `KE_ID_full` | gene set, key event or branch name |
| `gene_symbol` | HGNC gene symbol |

The mapping used in the published analysis is provided with the article as
Supplementary Table S1, and example files are given in `examples/`.

## What the application produces

- circular and aligned (flat) gene-set profiles, per sample and as multi-panel plates
- counts and proportions of significantly differentially expressed genes per gene set
- weighted cosine similarity scores between samples, with an optional restriction to
  the genes detected in all selected samples
- similarity matrices and hierarchical clustering
- silhouette widths per sample and per chemical
- paired mirrored comparisons of two samples, gene by gene

All figures and tables can be downloaded as PNG or CSV.

## Similarity score

Each sample is represented by a vector of log2 fold changes weighted by
`-log10(padj)`, with the weight clipped at a maximum value (**default 5**, as used in
the publication). The cosine of the angle between two such vectors is reported as the
cosine similarity score (CSS), ranging from -1 to +1.

The score is invariant to the overall scale of a profile: a high value indicates a
shared pattern of relative change, not responses of comparable amplitude.

## Licence

TODO: add a LICENSE file before release. Without one, the code is under default
copyright and cannot legally be reused.

## Contact

Emmanuel Demuynck — Vrije Universiteit Brussel / Sciensano
