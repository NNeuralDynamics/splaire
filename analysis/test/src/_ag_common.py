"""shared alphagenome scoring helpers — model loading, interval, indexing.

uses the all-chromosome alphagenome checkpoint (create_from_huggingface("all_folds")),
which is what both peer benchmarks (ESL_MPRA_2025, OpenSplice) used. 16,384 nt context
is the recommended window for variant effect prediction on splicing assays — OpenSplice
empirically validated that 16k beats 1M on splicing-variant correlation.

ontology_terms=None returns ALL tracks (4 SS + 734 SSU + 367 SJ for HOMO_SAPIENS,
as of alphagenome lib version pinned in alphagenome_env.yml).
ontology filtering is purely post-inference (no upstream compute cost).
"""
import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_ag_path = _here.parent.parent / "other_models" / "alphagenome_research"
sys.path.insert(0, str(_ag_path / "src"))

from alphagenome_research.model import dna_model  # noqa: E402
try:
    from alphagenome.models import dna_output
except ImportError:
    from alphagenome.data import dna_output
from alphagenome.data import genome  # noqa: E402


SS_N_CHANNELS = 4    # donor+, acceptor+, donor-, acceptor-
SSU_N_TRACKS = 734   # AG SPLICE_SITE_USAGE tissue tracks (HOMO_SAPIENS)
SJ_N_TRACKS = 367    # AG SPLICE_JUNCTIONS tissue tracks (HOMO_SAPIENS)

REQUESTED_OUTPUTS = [
    dna_output.OutputType.SPLICE_SITES,
    dna_output.OutputType.SPLICE_SITE_USAGE,
    dna_output.OutputType.SPLICE_JUNCTIONS,
]


def make_organism_settings(fasta_override=None, ag_data_dir=None):
    """build OrganismSettings from $AG_DATA_DIR. None if neither env var nor fasta_override."""
    if ag_data_dir is None:
        ag_data_dir = os.environ.get("AG_DATA_DIR")
    if not ag_data_dir and not fasta_override:
        return None
    if ag_data_dir:
        d = Path(ag_data_dir)
        return {
            dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
                fasta_path=str(fasta_override or d / "GRCh38.p13.genome.fa"),
                gtf_feather_path=str(d / "gencode.v46.annotation.gtf.gz.feather"),
                pas_feather_path=str(d / "polyadb_human_v3_exon3_contiguous_gtfv46.feather"),
                splice_site_starts_feather_path=str(d / "gencode.v46.splice_sites_starts.feather"),
                splice_site_ends_feather_path=str(d / "gencode.v46.splice_sites_ends.feather"),
            ),
        }
    return {
        dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(fasta_path=str(fasta_override)),
    }


def load_model(model_name="all_folds", source="huggingface", organism_settings=None):
    """load the AG model. all_folds = single all-chromosome checkpoint (not a CV ensemble)."""
    if source == "huggingface":
        return dna_model.create_from_huggingface(model_name, organism_settings=organism_settings)
    if source == "kaggle":
        return dna_model.create_from_kaggle(model_name, organism_settings=organism_settings)
    raise ValueError(f"unknown source: {source}")


def build_interval(chrom, pos, strand, window):
    """genomic interval centered on variant. 0-based half-open."""
    center = int(pos) - 1
    start = center - window // 2
    end = start + window
    return genome.Interval(chrom, start, end, strand=strand)


def site_idx(genomic_pos_1based, strand, win_start, win_end):
    """0-based array index for a canonical exon boundary, with AG's -1 splice-site peak shift.

    AG's SS prediction peaks one base into the exon from the canonical boundary
    (validated by position-scan in ESL_MPRA_2025 SI_figures/AlphaGenome). for a
    1-based genomic boundary at `pos`:
      + strand: idx = pos - win_start - 2   (-1 for 1->0 + -1 for AG peak)
      - strand: idx = win_end - pos - 1     (interval flipped to pre-mRNA frame + AG peak)
    """
    if strand == "+":
        return int(genomic_pos_1based) - 1 - win_start - 1
    return win_end - int(genomic_pos_1based) - 1


def flip_to_genomic(arr, strand):
    """flip per-position array from pre-mRNA frame back to genomic frame for - strand."""
    if strand == "-":
        return arr[::-1]
    return arr


def junction_idx_to_genomic(j_idx, strand, window_len):
    """convert junction array index from pre-mRNA frame to genomic frame for - strand."""
    if strand == "-":
        return window_len - 1 - int(j_idx)
    return int(j_idx)
