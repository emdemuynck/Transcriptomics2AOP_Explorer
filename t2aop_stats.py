"""Descriptive statistics for DESeq2 results and cosine-similarity matrices."""
import numpy as np
import pandas as pd


def _unique_genes(df):
    # DESeq2 results should have one row per gene and sample. Avoid double counting.
    return df.drop_duplicates(subset=['gene_symbol'], keep='first').copy()


def summarize_deg_totals(samples, padj_thr=0.05, fc_thr=0.0):
    rows=[]
    for sample, df in samples.items():
        d=_unique_genes(df)
        fc=pd.to_numeric(d['log2FoldChange'], errors='coerce')
        p=pd.to_numeric(d['padj'], errors='coerce')
        valid=fc.notna() & p.notna()
        sig=valid & (p<=padj_thr) & (fc.abs()>fc_thr)
        rows.append(dict(sample=sample, n_genes=len(d), n_testable=int(valid.sum()),
                         n_deg_total=int(sig.sum()), n_up=int((sig & (fc>0)).sum()),
                         n_down=int((sig & (fc<0)).sum()),
                         pct_deg_of_testable=100*sig.sum()/valid.sum() if valid.sum() else np.nan))
    return pd.DataFrame(rows)


def summarize_magnitude(samples, mapping, sets, padj_thr=0.05, fc_thr=0.0,
                        significant_only=False):
    rows=[]
    genes_by_set={s:set(mapping.loc[mapping['gene_set']==s,'gene_symbol'].dropna()) for s in sets}
    for sample, df in samples.items():
        d=_unique_genes(df)
        for gene_set, genes in genes_by_set.items():
            sub=d[d['gene_symbol'].isin(genes)]
            fc=pd.to_numeric(sub['log2FoldChange'],errors='coerce')
            p=pd.to_numeric(sub['padj'],errors='coerce')
            valid=fc.notna()
            sig=valid & p.notna() & (p<=padj_thr) & (fc.abs()>fc_thr)
            chosen=sig if significant_only else valid
            values=fc[chosen].to_numpy(dtype=float)
            absolute=np.abs(values)
            rows.append(dict(sample=sample,gene_set=gene_set,n_mapped_genes=len(genes),
                n_genes_with_fc=int(valid.sum()),n_deg=int(sig.sum()),n_used=len(values),
                mean_log2fc=float(np.mean(values)) if len(values) else np.nan,
                median_log2fc=float(np.median(values)) if len(values) else np.nan,
                mean_abs_log2fc=float(np.mean(absolute)) if len(values) else np.nan,
                median_abs_log2fc=float(np.median(absolute)) if len(values) else np.nan,
                p90_abs_log2fc=float(np.percentile(absolute,90)) if len(values) else np.nan,
                max_abs_log2fc=float(np.max(absolute)) if len(values) else np.nan,
                rms_log2fc=float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
                n_up_used=int((values>0).sum()),n_down_used=int((values<0).sum())))
    return pd.DataFrame(rows)


def summarize_css_pairs(sim, separator='_', chemical_level=1, percentiles=(75,90,95),
                        include_same_chemical=False):
    names=list(sim.index)
    records=[]
    for i, a in enumerate(names):
        for j in range(i+1,len(names)):
            b=names[j]
            ca=separator.join(str(a).split(separator)[:chemical_level])
            cb=separator.join(str(b).split(separator)[:chemical_level])
            if not include_same_chemical and ca==cb:
                continue
            val=pd.to_numeric(sim.loc[a,b],errors='coerce')
            if pd.isna(val):
                continue
            ca,cb=sorted((ca,cb))
            records.append(dict(compound_A=ca,compound_B=cb,sample_A=a,sample_B=b,CSS=float(val)))
    raw=pd.DataFrame(records,columns=['compound_A','compound_B','sample_A','sample_B','CSS'])
    columns=['compound_A','compound_B','n_comparisons','mean_CSS','median_CSS','std_CSS',
             'min_CSS','max_CSS']+[f'P{p}_CSS' for p in percentiles]
    if raw.empty:
        return pd.DataFrame(columns=columns),raw
    summary=[]
    for (a,b), sub in raw.groupby(['compound_A','compound_B'],sort=True):
        v=sub['CSS'].to_numpy(dtype=float)
        row=dict(compound_A=a,compound_B=b,n_comparisons=len(v),mean_CSS=np.mean(v),
                 median_CSS=np.median(v),std_CSS=np.std(v,ddof=1) if len(v)>1 else np.nan,
                 min_CSS=np.min(v),max_CSS=np.max(v))
        for p in percentiles:
            row[f'P{p}_CSS']=np.percentile(v,p)
        summary.append(row)
    return pd.DataFrame(summary)[columns].sort_values('P90_CSS',ascending=False),raw
