#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as mpl
import pathlib
import pysam
import re

from common.ref_func_corgi import HG38_SIZES, HG19_SIZES, T2T11_SIZES, TELOGATOR_SIZES, LEXICO_2_IND, makedir

REF_CHAR = 'MX=D'

REFFILE_NAMES = {'hg38':      HG38_SIZES,
                 'hg19':      HG19_SIZES,
                 't2t':       T2T11_SIZES,
                 'telogator': TELOGATOR_SIZES}

CYTOBAND_COLORS = {'gneg':(255,255,255),
                   'gpos25':(225,204,230),
                   'gpos50':(175,119,187),
                   'gpos75':(124,68,136),
                   'gpos100':(90,55,103),
                   'acen':(139,112,144),
                   'stalk':(139,112,144),
                   'gvar':(231,214,234)}


def strip_polymerase_coords(rn):
    return '/'.join(rn.split('/')[:-1])


def reads_2_cov(my_chr, readpos_list_all, out_dir, CONTIG_SIZES, WINDOW_SIZE):
    #
    if my_chr not in CONTIG_SIZES:
        print('skipping coverage computation for '+my_chr+'...')
        return None
    #
    print('computing coverage on '+my_chr+'...')
    cov = np.zeros(CONTIG_SIZES[my_chr], dtype='<i4')
    # collapse overlapping alignments
    for readpos_list in readpos_list_all:
        if len(readpos_list_all) == 0:
            continue
        found_overlaps = True
        while found_overlaps:
            found_overlaps = False
            for i in range(len(readpos_list)):
                for j in range(i+1,len(readpos_list)):
                    (x1, x2) = readpos_list[i]
                    (y1, y2) = readpos_list[j]
                    if x1 <= y2 and y1 <= x2:
                        found_overlaps = True
                        readpos_list[i] = (min([x1,y1]), max([x2,y2]))
                        del readpos_list[j]
                        break
                if found_overlaps:
                    break
        for rspan in readpos_list:
            cov[rspan[0]:rspan[1]] += 1
    # downsample
    out_cov = []
    for i in range(0,len(cov),WINDOW_SIZE):
        out_cov.append(np.mean(cov[i:i+WINDOW_SIZE]))
    cov = np.array(out_cov)
    #
    return cov


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='plot_coverage.py')
    parser.add_argument('-i',  type=str, required=True,  metavar='<str>', help="* input.bam")
    parser.add_argument('-o',  type=str, required=True,  metavar='<str>', help="* output_dir/")
    parser.add_argument('-r',  type=str, required=False, metavar='<str>', help="refname: hg38/hg19/t2t/telogator)", default='hg38')
    parser.add_argument('-q',  type=int, required=False, metavar='<int>', help="minimum MAPQ",                      default=0)
    parser.add_argument('-w',  type=int, required=False, metavar='<int>', help="window size for downsampling",      default=10000)
    parser.add_argument('-rt', type=str, required=False, metavar='<str>', help="read type: CCS/CLR/ONT",            default='CCS')
    args = parser.parse_args()

    IN_BAM = args.i

    READ_MODE = args.rt
    if READ_MODE not in ['CCS', 'CLR', 'ONT']:
        print('Error: -rt must be either CCS, CLR, or ONT')
        exit(1)

    OUT_DIR = args.o
    if OUT_DIR[-1] != '/':
        OUT_DIR += '/'
    makedir(OUT_DIR)
    OUT_NPZ = OUT_DIR+'cov.npz'

    MIN_MAPQ = max(0,args.q)
    WINDOW_SIZE = max(1,args.w)

    REF_VERS = args.r
    if REF_VERS not in REFFILE_NAMES:
        print('Error: Refname (-r) must be one of the following:')
        print(sorted(REFFILE_NAMES.keys()))
        exit(1)
    CONTIG_SIZES = REFFILE_NAMES[REF_VERS]

    sim_path = str(pathlib.Path(__file__).resolve().parent)
    resource_dir = sim_path + '/resources/'
    CYTOBAND_BED = resource_dir + f'{REF_VERS}-cytoband.bed'
    beddat_by_chr = {}
    with open(CYTOBAND_BED,'r') as f:
        for line in f:
            splt = line.strip().split('\t')
            if splt[0] not in beddat_by_chr:
                beddat_by_chr[splt[0]] = []
            beddat_by_chr[splt[0]].append((int(splt[1]), int(splt[2]), splt[3], splt[4]))

    prev_ref = None
    rnm_dict = {}
    alns_by_zmw = []    # alignment start/end per zmw
    rlen_by_zmw = []    # max tlen observed for each zmw
    covdat_by_ref = {}  #

    if IN_BAM[-4:].lower() == '.bam':
        #
        samfile = pysam.AlignmentFile(IN_BAM, "rb")
        refseqs = samfile.references
        #
        for aln in samfile.fetch(until_eof=True):
            splt = str(aln).split('\t')
            my_ref_ind  = splt[2].replace('#','')
            # pysam weirdness
            if my_ref_ind.isdigit():
                splt[2] = refseqs[int(my_ref_ind)]
            elif my_ref_ind == '-1':
                splt[2] = refseqs[-1]
            else:
                splt[2] = my_ref_ind
            #
            ref   = splt[2]
            pos   = int(splt[3])
            mapq  = int(splt[4])
            cigar = splt[5]

            if ref == '*':      # skip unmapped reads
                continue
            if mapq < MIN_MAPQ:
                continue
            #if pos > 1000000:  # for debugging purposes
            #   continue

            if READ_MODE == 'CLR':
                rnm = strip_polymerase_coords(splt[0])
                template_len = splt[0].split('/')[-1].split('_')
                template_len = int(template_len[1]) - int(template_len[0])
            elif READ_MODE in ['CCS', 'ONT']:
                rnm = splt[0]
                template_len = len(splt[9])

            if ref != prev_ref:
                # compute coverage on previous ref now that we're done
                if prev_ref is not None and len(alns_by_zmw) and prev_ref in CONTIG_SIZES:
                    covdat_by_ref[prev_ref] = reads_2_cov(prev_ref, alns_by_zmw, OUT_DIR, CONTIG_SIZES, WINDOW_SIZE)
                # reset for next ref
                if ref in CONTIG_SIZES:
                    print('processing reads on '+ref+'...')
                else:
                    print('skipping reads on '+ref+'...')
                alns_by_zmw = []
                rnm_dict = {}
                prev_ref = ref

            if ref not in CONTIG_SIZES:
                continue

            letters = re.split(r"\d+",cigar)[1:]
            numbers = [int(n) for n in re.findall(r"\d+",cigar)]
            adj     = 0
            for i in range(len(letters)):
                if letters[i] in REF_CHAR:
                    adj += numbers[i]

            if rnm in rnm_dict:
                my_rind = rnm_dict[rnm]
            else:
                rnm_dict[rnm] = len(rnm_dict)
                my_rind       = len(rnm_dict)-1
                alns_by_zmw.append([])
                rlen_by_zmw.append(0)

            alns_by_zmw[my_rind].append((pos, pos+adj))
            rlen_by_zmw[my_rind] = max([rlen_by_zmw[my_rind], template_len])
        samfile.close()

        # we probably we need to process the final ref, assuming no contigs beyond chrM
        if ref not in covdat_by_ref and len(alns_by_zmw) and ref in CONTIG_SIZES:
            covdat_by_ref[ref] = reads_2_cov(ref, alns_by_zmw, OUT_DIR, CONTIG_SIZES, WINDOW_SIZE)

        # save output
        sorted_chr = [n[1] for n in sorted([(LEXICO_2_IND[k],k) for k in covdat_by_ref.keys()])]
        np.savez_compressed(OUT_NPZ, sorted_chr=sorted_chr, **covdat_by_ref)
    #
    elif IN_BAM[-4:].lower() == '.npz':
        in_npz = np.load(IN_BAM)
        sorted_chr = in_npz['sorted_chr']
        covdat_by_ref = {k:in_npz[k] for k in sorted_chr}
    #
    else:
        print('Error: -i must be .bam or .npz')
        exit(1)

    print(sorted_chr)


if __name__ == '__main__':
    main()
