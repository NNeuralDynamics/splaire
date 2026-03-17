"""sequence and splice annotation encoding for ml"""

import numpy as np
from math import ceil
from constants import CL_max, SL

assert CL_max % 2 == 0

# one-hot encoding for basic mode: 0=pad/N, 1=A, 2=C, 3=G, 4=T
IN_MAP_BASIC = np.asarray([
    [0, 0, 0, 0],  # 0 = pad/N
    [1, 0, 0, 0],  # 1 = A
    [0, 1, 0, 0],  # 2 = C
    [0, 0, 1, 0],  # 3 = G
    [0, 0, 0, 1],  # 4 = T
], dtype='float32')

# one-hot encoding for het mode: includes IUPAC codes for heterozygous sites
# 0=N, 1=A, 2=C, 3=G, 4=T, 5=M(A/C), 6=R(A/G), 7=W(A/T), 8=S(C/G), 9=Y(C/T), 10=K(G/T)
IN_MAP_HET = np.asarray([
    [0.0, 0.0, 0.0, 0.0],  # 0 = N/padding
    [1.0, 0.0, 0.0, 0.0],  # 1 = A
    [0.0, 1.0, 0.0, 0.0],  # 2 = C
    [0.0, 0.0, 1.0, 0.0],  # 3 = G
    [0.0, 0.0, 0.0, 1.0],  # 4 = T
    [0.5, 0.5, 0.0, 0.0],  # 5 = M (A/C)
    [0.5, 0.0, 0.5, 0.0],  # 6 = R (A/G)
    [0.5, 0.0, 0.0, 0.5],  # 7 = W (A/T)
    [0.0, 0.5, 0.5, 0.0],  # 8 = S (C/G)
    [0.0, 0.5, 0.0, 0.5],  # 9 = Y (C/T)
    [0.0, 0.0, 0.5, 0.5],  # 10 = K (G/T)
], dtype='float32')

# structured dtype for genomic coordinates
GC_DTYPE = np.dtype([
    ('chrom', 'int8'),
    ('strand', 'int8'),
    ('position', 'int32'),
    ('name', 'S200')
])

# character to index mappings
CHAR_TO_IDX_BASIC = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
CHAR_TO_IDX_HET = {
    'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4,
    'M': 5, 'R': 6, 'W': 7, 'S': 8, 'Y': 9, 'K': 10
}

# IUPAC codes for heterozygous allele pairs
IUPAC_HET = {
    ('A', 'C'): 'M', ('C', 'A'): 'M',
    ('A', 'G'): 'R', ('G', 'A'): 'R',
    ('A', 'T'): 'W', ('T', 'A'): 'W',
    ('C', 'G'): 'S', ('G', 'C'): 'S',
    ('C', 'T'): 'Y', ('T', 'C'): 'Y',
    ('G', 'T'): 'K', ('T', 'G'): 'K',
}

# complement mapping for reverse strand
BASE_COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def ceil_div(x, y):
    return int(ceil(float(x) / y))


def parse_int_list(x):
    if not x:
        return []
    return [int(v) for v in x.split(',') if v.strip()]


def parse_float_list(x):
    if not x:
        return []
    return [float(v) for v in x.split(',') if v.strip()]


def normalize_chrom(chrom):
    if isinstance(chrom, str) and chrom.startswith('chr'):
        chrom = chrom[3:]
    if chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    return int(chrom)


def seq_to_int_array(seq, mode='basic'):
    char_map = CHAR_TO_IDX_HET if mode == 'het' else CHAR_TO_IDX_BASIC
    return np.array([char_map.get(ch, 0) for ch in seq.upper()], dtype='int8')


def reverse_complement_int(x0, mode='basic'):
    if mode == 'basic':
        return (5 - x0[::-1]) % 5

    # het mode: build complement map including IUPAC codes
    # M(AC)->K(GT), R(AG)->Y(CT), W(AT)->W(AT), S(CG)->S(CG), Y(CT)->R(AG), K(GT)->M(AC)
    comp_map = {
        0: 0,   # N -> N
        1: 4,   # A -> T
        2: 3,   # C -> G
        3: 2,   # G -> C
        4: 1,   # T -> A
        5: 10,  # M (A/C) -> K (G/T)
        6: 9,   # R (A/G) -> Y (C/T)
        7: 7,   # W (A/T) -> W (A/T)
        8: 8,   # S (C/G) -> S (C/G)
        9: 6,   # Y (C/T) -> R (A/G)
        10: 5,  # K (G/T) -> M (A/C)
    }
    return np.array([comp_map[c] for c in x0[::-1]], dtype='int8')


def one_hot_encode(xd, yd, mode='basic'):
    in_map = IN_MAP_HET if mode == 'het' else IN_MAP_BASIC
    return in_map[xd.astype('int8')], yd


def reformat_data(x0, y0, gc_y=None):
    n_dims = y0.shape[1] if y0.ndim > 1 else 4
    num_points = ceil_div(len(y0), SL)

    # pad x0 and y0 to fit blocks
    x0 = np.pad(x0, (0, SL + CL_max), 'constant', constant_values=0)
    y0 = np.pad(y0, ((0, SL), (0, 0)), mode='constant', constant_values=0)

    # allocate block arrays
    xd = np.zeros((num_points, SL + CL_max), dtype='int8')
    yd = np.zeros((num_points, SL, n_dims), dtype='float32')

    for i in range(num_points):
        xd[i] = x0[i * SL: i * SL + SL + CL_max]
        yd[i] = y0[i * SL: (i + 1) * SL]

    # handle genomic coordinates if provided
    gcd = None
    if gc_y is not None:
        gc_y = gc_y + [(-1, -1, -1, 'padding')] * (SL - len(gc_y) % SL)
        gc_struct = np.array(gc_y, dtype=GC_DTYPE)
        gcd = np.zeros((num_points, SL), dtype=GC_DTYPE)
        for i in range(num_points):
            gcd[i] = gc_struct[i * SL: (i + 1) * SL]

    return xd, yd, gcd


def create_datapoints(
    seq,
    strand,
    tx_start,
    tx_end,
    jn_start,
    jn_end,
    jn_start_sse,
    jn_end_sse,
    chrom,
    name,
    remove_missing=True,
    mode='basic',
    jn_start_sse_pop=None,
    jn_end_sse_pop=None,
    jn_start_sse_delta=None,
    jn_end_sse_delta=None,
    var_pos=None,
    var_ref=None,
    var_alt=None,
    var_bin=None,
):
    seq = seq.upper()
    tx_start = int(tx_start)
    tx_end = int(tx_end)
    chrom_int = normalize_chrom(chrom)

    # parse junction positions and SSU values
    jn_start_pos = parse_int_list(jn_start[0]) if jn_start else []
    jn_end_pos = parse_int_list(jn_end[0]) if jn_end else []
    jn_start_ssu = parse_float_list(jn_start_sse[0]) if jn_start_sse else []
    jn_end_ssu = parse_float_list(jn_end_sse[0]) if jn_end_sse else []

    # for pop mode, also parse population SSU and delta
    if mode == 'pop':
        jn_start_pop = parse_float_list(jn_start_sse_pop[0]) if jn_start_sse_pop else []
        jn_end_pop = parse_float_list(jn_end_sse_pop[0]) if jn_end_sse_pop else []
        jn_start_delta = parse_float_list(jn_start_sse_delta[0]) if jn_start_sse_delta else []
        jn_end_delta = parse_float_list(jn_end_sse_delta[0]) if jn_end_sse_delta else []

    # validate lengths
    assert len(jn_start_pos) == len(jn_start_ssu), "jn_start length mismatch"
    assert len(jn_end_pos) == len(jn_end_ssu), "jn_end length mismatch"

    # add padding to sequence
    pad = CL_max // 2
    seq_padded = 'N' * pad + seq + 'N' * pad

    # for het mode, annotate heterozygous sites before encoding
    if mode == 'het' and var_pos is not None:
        seq_list = list(seq_padded)
        for pos, ref, alt, bin_flag in zip(var_pos, var_ref, var_alt, var_bin):
            if bin_flag != 1:
                continue
            # compute relative index in padded sequence
            if strand == '+':
                rel = (pos - tx_start) + pad
            else:
                rel = (tx_end - pos) + pad
            if 0 <= rel < len(seq_list):
                pair = (ref.upper(), alt.upper())
                if pair in IUPAC_HET:
                    seq_list[rel] = IUPAC_HET[pair]
        seq_padded = ''.join(seq_list)

    # convert to integer array
    x0 = seq_to_int_array(seq_padded, mode=mode)

    # initialize label array
    length = tx_end - tx_start + 1
    n_dims = 6 if mode == 'pop' else 4
    y0 = np.zeros((length, n_dims), dtype='float32')
    # default to "neither" class
    y0[:, 0] = 1.0

    # build genomic coordinates list
    gc_y = []

    # helper to set label at position
    def set_label(idx, class_vec, ssu, pop_ssu=0.0, delta=0.0):
        if remove_missing and ssu == 777:
            y0[idx] = [1, 0, 0, ssu] if n_dims == 4 else [1, 0, 0, ssu, pop_ssu, delta]
        else:
            if n_dims == 4:
                y0[idx] = class_vec + [ssu]
            else:
                y0[idx] = class_vec + [ssu, pop_ssu, delta]

    # fill in splice site labels
    # on + strand: jn_start = donor (exon end), jn_end = acceptor (exon start)
    # on - strand: swap roles and reverse positions
    if strand == '+':
        # donors from jn_start
        for i, pos in enumerate(jn_start_pos):
            if tx_start <= pos <= tx_end:
                idx = pos - tx_start
                if mode == 'pop':
                    set_label(idx, [0, 0, 1], jn_start_ssu[i], jn_start_pop[i], jn_start_delta[i])
                else:
                    set_label(idx, [0, 0, 1], jn_start_ssu[i])
        # acceptors from jn_end
        for i, pos in enumerate(jn_end_pos):
            if tx_start <= pos <= tx_end:
                idx = pos - tx_start
                if mode == 'pop':
                    set_label(idx, [0, 1, 0], jn_end_ssu[i], jn_end_pop[i], jn_end_delta[i])
                else:
                    set_label(idx, [0, 1, 0], jn_end_ssu[i])
        # genomic coordinates for + strand
        for i in range(tx_start, tx_end + 1):
            gc_y.append((chrom_int, 1, i, name))
    else:
        # reverse complement the sequence
        x0 = reverse_complement_int(x0, mode=mode)
        # on - strand: jn_end becomes donor, jn_start becomes acceptor
        # positions are mirrored around transcript
        for i, pos in enumerate(jn_end_pos):
            if tx_start <= pos <= tx_end:
                idx = tx_end - pos
                if mode == 'pop':
                    set_label(idx, [0, 0, 1], jn_end_ssu[i], jn_end_pop[i], jn_end_delta[i])
                else:
                    set_label(idx, [0, 0, 1], jn_end_ssu[i])
        for i, pos in enumerate(jn_start_pos):
            if tx_start <= pos <= tx_end:
                idx = tx_end - pos
                if mode == 'pop':
                    set_label(idx, [0, 1, 0], jn_start_ssu[i], jn_start_pop[i], jn_start_delta[i])
                else:
                    set_label(idx, [0, 1, 0], jn_start_ssu[i])
        # genomic coordinates for - strand (reversed)
        for i in range(tx_end, tx_start - 1, -1):
            gc_y.append((chrom_int, 0, i, name))

    # split into blocks
    xd, yd, gcd = reformat_data(x0, y0, gc_y)

    # one-hot encode
    x_encoded, y_encoded = one_hot_encode(xd, yd, mode=mode)

    return x_encoded, y_encoded, gcd
