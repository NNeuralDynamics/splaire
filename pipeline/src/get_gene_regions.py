#!/usr/bin/env python3
"""extract non-overlapping protein-coding gene regions from GTF"""

import pandas as pd
import logging
import argparse

def parse_gtf(gtf_path):
    cols = ['Chromosome','source','type','Start','End','score','Strand','phase','Attributes']
    df = pd.read_csv(
        gtf_path, sep='\t', comment='#', header=None, names=cols,
        usecols=['Chromosome','type','Start','End','Strand','Attributes'],
        dtype={'Chromosome':str}
    )
    df = df[
        (df['type']=='transcript') &
        df['Attributes'].str.contains('transcript_type "protein_coding"')
    ]
    df['Gene_ID'] = df['Attributes'].str.extract('gene_id "([^"]+)"')
    return df[['Chromosome','Start','End','Strand','Gene_ID']]

def consolidate(df):
    return (
        df.groupby('Gene_ID', sort=False)
          .agg(Chromosome=('Chromosome','first'),
               Start      =('Start','min'),
               End        =('End','max'),
               Strand     =('Strand','first'))
          .reset_index()
    )

def resolve_overlaps(df):
    changed = False
    out = []
    for (chrom, strand), grp in df.groupby(['Chromosome','Strand'], sort=False):
        grp = grp.sort_values('Start').reset_index(drop=True)
        acc = []
        for _, cur in grp.iterrows():
            cur = cur.copy()
            if acc and cur.Start < acc[-1].End:
                prev = acc[-1]
                # complete containment?
                if cur.Start >= prev.Start and cur.End <= prev.End:
                    changed = True
                    continue
                ov = prev.End - cur.Start
                p_len = prev.End - prev.Start
                c_len = cur.End  - cur.Start
                if ov/p_len <= ov/c_len:
                    new_end = cur.Start
                    if new_end > prev.Start:
                        acc[-1].End = new_end
                        changed = True
                    else:
                        acc.pop()
                        changed = True
                else:
                    new_start = prev.End
                    if new_start < cur.End:
                        cur.Start = new_start
                        changed = True
                    else:
                        changed = True
                        continue
            acc.append(cur)
        out.append(pd.DataFrame(acc))
    return pd.concat(out, ignore_index=True), changed

def main(gtf, out_tsv, out_bed, log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    df = parse_gtf(gtf)
    df = consolidate(df)

    passes = 0
    changed = True
    while changed:
        df, changed = resolve_overlaps(df)
        passes += 1
        logging.info(f"Pass {passes}, changed={changed}")
    logging.info(f"Resolved all overlaps in {passes} passes")

    df.to_csv(out_tsv, sep='\t', index=False)
    df[['Chromosome','Start','End','Gene_ID','Strand']]\
      .to_csv(out_bed, sep='\t', index=False, header=False)
    print(f"Done: {len(df)} genes, {passes} passes.")

if __name__=='__main__':
    p = argparse.ArgumentParser(description="Extract non‑overlapping protein‑coding genes")
    p.add_argument('-g','--gtf', required=True, help="Input GTF")
    p.add_argument('-t','--tsv', required=True, help="Output TSV")
    p.add_argument('-b','--bed', required=True, help="Output BED")
    p.add_argument('-l','--log', required=True, help="Log file")
    args = p.parse_args()
    main(args.gtf, args.tsv, args.bed, args.log)

