#!/usr/bin/env python
import argparse
import bisect
import gzip
import numpy as np
import math
import matplotlib.pyplot as mpl
import pathlib
import pysam
import re
import sys
import time

from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from scipy.stats import wasserstein_distance

from common.bimodal_gaussian import fit_bimodal_gaussian, plot_bimodal_fit
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
CYTOBAND_COLORS = {k:(v[0]/255.,v[1]/255.,v[2]/255.) for k,v in CYTOBAND_COLORS.items()}

CHROM_COLOR_CYCLE = [(32, 119, 184),
                     (245, 126, 19),
                     (43, 159, 44),
                     (214, 37, 42),
                     (143, 103, 182),
                     (139, 85, 78),
                     (226, 120, 194),
                     (126, 126, 126),
                     (189, 188, 36),
                     (27, 189, 210)]
CHROM_COLOR_CYCLE = [(v[0]/255.,v[1]/255.,v[2]/255.) for v in CHROM_COLOR_CYCLE]

# mask these regions when computing average coverage
UNSTABLE_REGION = ['acen', 'gvar', 'stalk']
UNSTABLE_CHR = ['chrM']

TWO_PI = 2.0*np.pi

COV_YT = range(-3,3+1)
COV_YL = [str(n) for n in COV_YT]
KDE_NUMPOINTS_VAF = 50
KDE_STD_VAF = 0.025*KDE_NUMPOINTS_VAF
KDE_STD_POS = 20000
KDE_YT = [0.0, 0.25*KDE_NUMPOINTS_VAF, 0.50*KDE_NUMPOINTS_VAF, 0.75*KDE_NUMPOINTS_VAF, KDE_NUMPOINTS_VAF]
KDE_YL = ['0%', '25%', '50%', '75%', '100%']
KDE_YL = ['0', '.25', '.50', '.75', '1']


def strip_polymerase_coords(rn):
    return '/'.join(rn.split('/')[:-1])


def reads_2_cov(my_chr, readpos_list_all, out_dir, CONTIG_SIZES, WINDOW_SIZE, bed_regions):
    #
    if my_chr not in CONTIG_SIZES:
        print(' - skipping coverage computation for '+my_chr+'...')
        return None
    #
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
    # report coverage for specific bed regions
    bed_out = []
    if my_chr in bed_regions:
        for br in bed_regions[my_chr]:
            b1 = max(br[0],0)
            b2 = min(br[1],len(cov))
            bed_out.append([br, np.mean(cov[b1:b2]), np.median(cov[b1:b2]), np.std(cov[b1:b2])])
    # downsample
    if WINDOW_SIZE <= 1:
        return (cov, bed_out)
    out_cov = []
    for i in range(0,len(cov),WINDOW_SIZE):
        out_cov.append(np.mean(cov[i:i+WINDOW_SIZE]))
    cov = np.array(out_cov)
    return (cov, bed_out)


def log_px(x, y, ux, uy, ox, oy):
    out = -0.5*np.log(TWO_PI*ox) - ((x-ux)*(x-ux))/(2*ox*ox) - 0.5*np.log(TWO_PI*oy) - ((y-uy)*(y-uy))/(2*oy*oy)
    return out


def find_indices_in_range(sorted_list, lb, ub):
    start_idx = bisect.bisect_left(sorted_list, lb)
    end_idx = bisect.bisect_right(sorted_list, ub)
    return (start_idx, end_idx)


def emd(pdf_a, pdf_b):
    return wasserstein_distance(np.arange(len(pdf_a)), np.arange(len(pdf_b)), pdf_a, pdf_b)


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='plot_coverage.py')
    parser.add_argument('-i',  type=str, required=True,  metavar='<str>',     help="* input.bam")
    parser.add_argument('-o',  type=str, required=True,  metavar='<str>',     help="* output_dir/")
    parser.add_argument('-r',  type=str, required=False, metavar='<str>',     help="refname: t2t / hg38 / hg19",          default='t2t')
    parser.add_argument('-q',  type=int, required=False, metavar='<int>',     help="minimum MAPQ",                        default=0)
    parser.add_argument('-w',  type=int, required=False, metavar='<int>',     help="window size for downsampling",        default=10000)
    parser.add_argument('-b',  type=str, required=False, metavar='<str>',     help="bed of regions to query",             default='')
    parser.add_argument('-v',  type=str, required=False, metavar='<str>',     help="input.vcf",                           default='')
    parser.add_argument('-vw', type=str, required=False, metavar='<str>',     help="variant filters (whitelist)",         default='PASS,germline')
    parser.add_argument('-vb', type=str, required=False, metavar='<str>',     help="variant filters (blacklist)",         default='weak_evidence')
    parser.add_argument('-vd', type=int, required=False, metavar='<int>',     help="window size for var density",         default=10000)
    parser.add_argument('-cw', type=int, required=False, metavar='<int>',     help="window size for cnv prediction",      default=1000000)
    parser.add_argument('-cv', type=int, required=False, metavar='<int>',     help="minimum variants for cnv prediction", default=50)
    parser.add_argument('-s',  type=str, required=False, metavar='<str>',     help="sample name",                         default='')
    parser.add_argument('-rt', type=str, required=False, metavar='<str>',     help="read type: CCS / CLR / ONT",          default='CCS')
    parser.add_argument('--report-cnvs', required=False, action='store_true', help="[EXPERIMENTAL] report CNVs",          default=False)
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
    PLOT_DIR = OUT_DIR + 'plots/'
    makedir(PLOT_DIR)

    MIN_MAPQ = max(0, args.q)
    WINDOW_SIZE = max(1, args.w)
    VAR_WINDOW = max(1, args.vd)
    CNV_WINDOW = max(1, args.cw)
    CNV_MINVAR = max(1, args.cv)
    CNV_NUM_INDS = int((CNV_WINDOW / VAR_WINDOW) + 0.5)

    HOM_VAF_THRESH = 0.900

    REF_VERS  = args.r
    BED_FILE  = args.b
    IN_VCF    = args.v
    SAMP_NAME = args.s

    VAR_FILT_WHITELIST = args.vw.split(',')
    VAR_FILT_BLACKLIST = args.vb.split(',')

    REPORT_COPYNUM = args.report_cnvs

    OUT_NPZ = f'{OUT_DIR}cov.npz'
    VAF_NPZ = f'{OUT_DIR}vaf.npz'
    CNV_BED = f'{OUT_DIR}cnv.bed'
    if SAMP_NAME:
        OUT_NPZ = f'{OUT_DIR}cov_{SAMP_NAME}.npz'
        VAF_NPZ = f'{OUT_DIR}vaf_{SAMP_NAME}.npz'
        CNV_BED = f'{OUT_DIR}cnv_{SAMP_NAME}.bed'

    bed_regions = {}
    if BED_FILE:
        with open(BED_FILE,'r') as f:
            for line in f:
                splt = line.strip().split('\t')
                if len(splt) < 3: # malformed or empty line
                    continue
                if len(splt) >= 4:
                    bed_annot = ','.join(splt[3:])
                else:
                    bed_annot = ''
                if splt[0] not in bed_regions:
                    bed_regions[splt[0]] = []
                (p1, p2) = sorted([int(splt[1]), int(splt[2])])
                bed_regions[splt[0]].append((p1, p2, bed_annot))

    sim_path = str(pathlib.Path(__file__).resolve().parent)
    resource_dir = sim_path + '/resources/'
    CYTOBAND_BED = resource_dir + f'{REF_VERS}-cytoband.bed'
    cyto_by_chr = {}
    unstable_by_chr = {}
    with open(CYTOBAND_BED,'r') as f:
        for line in f:
            splt = line.strip().split('\t')
            if splt[0] not in cyto_by_chr:
                cyto_by_chr[splt[0]] = []
                unstable_by_chr[splt[0]] = []
            cyto_by_chr[splt[0]].append((int(splt[1]), int(splt[2]), splt[3], splt[4]))
            if splt[4] in UNSTABLE_REGION:
                unstable_by_chr[splt[0]].append((int(splt[1]), int(splt[2])))
    BUFFER_UNSTABLE = max(int(1000000/VAR_WINDOW), 1)

    prev_ref = None
    rnm_dict = {}
    alns_by_zmw = []    # alignment start/end per zmw
    rlen_by_zmw = []    # max tlen observed for each zmw
    covdat_by_ref = {}  #
    all_bed_result = []
    tt = time.perf_counter()

    if IN_BAM[-4:].lower() == '.bam':
        #
        if REF_VERS not in REFFILE_NAMES:
            print('Error: -r must be one of the following:')
            print(sorted(REFFILE_NAMES.keys()))
            exit(1)
        CONTIG_SIZES = REFFILE_NAMES[REF_VERS]
        print(f'using reference: {REF_VERS}')
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
                    (covdat_by_ref[prev_ref], bed_results) = reads_2_cov(prev_ref, alns_by_zmw, OUT_DIR, CONTIG_SIZES, WINDOW_SIZE, bed_regions)
                    all_bed_result.extend(bed_results)
                    sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
                    sys.stdout.flush()
                # reset for next ref
                if ref in CONTIG_SIZES:
                    sys.stdout.write(f'processing reads on {ref}...')
                    sys.stdout.flush()
                    tt = time.perf_counter()
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
            (covdat_by_ref[ref], bed_results) = reads_2_cov(ref, alns_by_zmw, OUT_DIR, CONTIG_SIZES, WINDOW_SIZE, bed_regions)
            all_bed_result.extend(bed_results)
        sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
        sys.stdout.flush()

        # save output
        sorted_chr = [n[1] for n in sorted([(LEXICO_2_IND[k],k) for k in covdat_by_ref.keys()])]
        np.savez_compressed(OUT_NPZ, ref_vers=REF_VERS, window_size=WINDOW_SIZE, sorted_chr=sorted_chr, **covdat_by_ref)
    #
    elif IN_BAM[-4:].lower() == '.npz':
        print('reading from an existing npz archive instead of bam...')
        in_npz = np.load(IN_BAM)
        REF_VERS = str(in_npz['ref_vers'])
        WINDOW_SIZE = int(in_npz['window_size'])
        sorted_chr = in_npz['sorted_chr'].tolist()
        covdat_by_ref = {k:in_npz[k] for k in sorted_chr}
        print(f' - ignoring -r and instead using: {REF_VERS}')
        print(f' - ignoring -w and instead using: {WINDOW_SIZE}')
        CONTIG_SIZES = REFFILE_NAMES[REF_VERS]
        if BED_FILE:
            print('Warning: a bed file was specified but will be ignored because input is .npz')
    #
    else:
        print('Error: -i must be .bam or .npz')
        exit(1)

    #
    # READ VCFs
    # we're assuming vcfs have GT and AF fields, and are sorted
    #
    var_kde_by_chr = {}
    var_het_by_chr = {}
    var_hom_by_chr = {}
    USING_VAR_NPZ = False
    if IN_VCF:
        if IN_VCF[-4:].lower() == '.vcf' or IN_VCF[-7:].lower() == '.vcf.gz':
            sys.stdout.write('reading input VCF...')
            sys.stdout.flush()
            tt = time.perf_counter()
            is_gzipped = True
            with gzip.open(IN_VCF, 'r') as fh:
                try:
                    fh.read(1)
                except OSError:
                    is_gzipped = False
            if is_gzipped:
                f = gzip.open(IN_VCF,'rt')
            else:
                f = open(IN_VCF,'r')
            for line in f:
                if line[0] != '#':
                    splt = line.strip().split('\t')
                    my_chr = splt[0]
                    my_pos = int(splt[1])
                    my_filt = splt[6].split(';')
                    # filters of interest:
                    # -- Clair3 / ClairS-TO: PASS,  NonSomatic
                    # -- Mutect2: germline, haplotype, panel_of_normals, weak_evidence
                    if any(n in VAR_FILT_BLACKLIST for n in my_filt):
                        continue
                    if any(n in VAR_FILT_WHITELIST for n in my_filt):
                        fmt_split = splt[8].split(':')
                        dat_split = splt[9].split(':')
                        if 'GT' in fmt_split and 'AF' in fmt_split:
                            ind_gt = fmt_split.index('GT')
                            ind_af = fmt_split.index('AF')
                            my_gt = dat_split[ind_gt]
                            my_af = float(dat_split[ind_af].split(',')[0]) # if multiallelic, take first
                            if my_gt == '1/1' or my_gt == '1|1' or my_af >= HOM_VAF_THRESH:
                                if my_chr not in var_hom_by_chr:
                                    var_hom_by_chr[my_chr] = [[], []]
                                var_hom_by_chr[my_chr][0].append(my_pos)
                                var_hom_by_chr[my_chr][1].append(my_af)
                            else:
                                if my_chr not in var_het_by_chr:
                                    var_het_by_chr[my_chr] = [[], []]
                                var_het_by_chr[my_chr][0].append(my_pos)
                                var_het_by_chr[my_chr][1].append(my_af)
                                #print(splt[0], splt[1], splt[3], splt[4], my_filt, my_gt, my_af)
            f.close()
            sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
            sys.stdout.flush()
            #
            # keep the raw calls for CNV prediction
            #
            for my_chr in sorted_chr:
                if my_chr in var_het_by_chr:
                    var_het_by_chr[my_chr] = np.array(var_het_by_chr[my_chr])
                else:
                    var_het_by_chr[my_chr] = np.array([[],[]])
                if my_chr in var_hom_by_chr:
                    var_hom_by_chr[my_chr] = np.array(var_hom_by_chr[my_chr])
                else:
                    var_hom_by_chr[my_chr] = np.array([[],[]])
        #
        elif IN_VCF[-4:].lower() == '.npz':
            print('reading from an existing npz archive instead of vcf...')
            in_npz = np.load(IN_VCF)
            if in_npz['extra_covwin'] != WINDOW_SIZE:
                print('Error: coverage window size in variant npz does not match.')
                exit(1)
            VAR_WINDOW = int(in_npz['extra_varwin'])
            VAR_FILT_WHITELIST = str(in_npz['var_filt_whitelist']).split(',')
            VAR_FILT_BLACKLIST = str(in_npz['var_filt_blacklist']).split(',')
            print(f' - ignoring -vd and instead using: {VAR_WINDOW}')
            print(f' - ignoring -vw and instead using: {VAR_FILT_WHITELIST}')
            print(f' - ignoring -vb and instead using: {VAR_FILT_BLACKLIST}')
            var_kde_by_chr = {k:in_npz[f'kde_{k}'] for k in sorted_chr}
            var_het_by_chr = {k:in_npz[f'het_{k}'] for k in sorted_chr}
            var_hom_by_chr = {k:in_npz[f'hom_{k}'] for k in sorted_chr}
            USING_VAR_NPZ = True
        #
        else:
            print('Error: -v must be .vcf or .vcf.gz or .npz')
            exit(1)

    #
    # compute het/hom variant densities
    #
    het_dens_by_chr = {}
    hom_dens_by_chr = {}
    for my_chr in sorted_chr:
        my_dens = np.zeros((int(CONTIG_SIZES[my_chr]/VAR_WINDOW)+1), dtype='float')
        for my_vpos in var_het_by_chr[my_chr][0,:]:
            my_dens[int(my_vpos)//VAR_WINDOW] += 1.0
        het_dens_by_chr[my_chr] = np.array(my_dens, copy=True)
        #
        my_dens = np.zeros((int(CONTIG_SIZES[my_chr]/VAR_WINDOW)+1), dtype='float')
        for my_vpos in var_hom_by_chr[my_chr][0,:]:
            my_dens[int(my_vpos)//VAR_WINDOW] += 1.0
        hom_dens_by_chr[my_chr] = np.array(my_dens, copy=True)

    #
    # determine average coverage across whole genome
    # -- in most cases this will correspond to 2 copies, but not always
    #
    all_win = []
    masked_covdat_by_ref = {}
    for my_chr in sorted_chr:
        if my_chr in UNSTABLE_CHR:
            masked_covdat_by_ref[my_chr] = covdat_by_ref[my_chr]
            continue
        cy = np.copy(covdat_by_ref[my_chr])
        if my_chr in unstable_by_chr:
            for ur in unstable_by_chr[my_chr]:
                w1 = max(math.floor(ur[0]/WINDOW_SIZE) - BUFFER_UNSTABLE, 0)
                w2 = min(math.ceil(ur[1]/WINDOW_SIZE) + BUFFER_UNSTABLE, len(cy)-1)
                cy[w1:w2+1] = -1.0
        masked_covdat_by_ref[my_chr] = np.copy(cy)
        all_win.extend(cy[cy >= 0.0].tolist())
    all_avg_cov = (np.mean(all_win), np.median(all_win), np.std(all_win))
    avg_log2 = np.log2(np.median(all_win))
    #
    fig = mpl.figure(1, figsize=(10,5))
    with np.errstate(divide='ignore', invalid='ignore'):
        mpl.hist(np.log2(all_win) - avg_log2, bins=300, range=[COV_YT[0], COV_YT[-1]])
    mpl.xticks(COV_YT, COV_YL)
    mpl.xlim([COV_YT[0], COV_YT[-1]])
    mpl.grid(which='both', linestyle='--', alpha=0.6)
    mpl.xlabel('normalized log2 depth')
    mpl.ylabel('bin count')
    mpl.tight_layout()
    mpl.savefig(f'{PLOT_DIR}depth-hist.png')
    mpl.close(fig)
    del all_win

    #
    #
    #

    fig_width_scalar = 11.5 / CONTIG_SIZES['chr1']
    fig_width_buffer = 0.5
    fig_width_min    = 2.0
    fig_height       = 5.0
    #
    plotted_cx_cy = {}
    cnv_bed_out = []

    for my_chr in sorted_chr:
        sys.stdout.write(f'making plots for {my_chr}...')
        sys.stdout.flush()
        tt = time.perf_counter()
        #
        xt = np.arange(0,CONTIG_SIZES[my_chr],10000000)
        xl = [f'{n*10}M' for n in range(len(xt))]
        with np.errstate(divide='ignore', invalid='ignore'):
            #cy = np.log2(covdat_by_ref[my_chr]) - avg_log2
            cy = np.log2(masked_covdat_by_ref[my_chr]) - avg_log2
        cx = np.array([n*WINDOW_SIZE + WINDOW_SIZE/2 for n in range(len(cy))])
        plotted_cx_cy[my_chr] = (np.copy(cx), np.copy(cy))
        #
        if my_chr in UNSTABLE_CHR:
            print(f' skipping {my_chr}.')
            Z = np.zeros((KDE_NUMPOINTS_VAF, int(CONTIG_SIZES[my_chr]/WINDOW_SIZE)+1), dtype='float')
            var_kde_by_chr[my_chr] = np.array(Z, copy=True)
            continue

        #
        # PLOTTING
        #
        my_width = max(CONTIG_SIZES[my_chr]*fig_width_scalar + fig_width_buffer, fig_width_min)
        fig = mpl.figure(1, figsize=(my_width,fig_height), dpi=200)
        gs = gridspec.GridSpec(3, 1, height_ratios=[4,4,1])
        ax1 = mpl.subplot(gs[0])
        mpl.scatter(cx, cy, s=1, c='black')
        mpl.xlim(0, CONTIG_SIZES[my_chr])
        mpl.ylim(COV_YT[0], COV_YT[-1])
        mpl.xticks(xt,xl)
        mpl.yticks(COV_YT, COV_YL)
        mpl.grid(which='both', linestyle='--', alpha=0.6)
        mpl.ylabel('log2 cov change')
        for tick in ax1.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        #
        ax2 = mpl.subplot(gs[1])
        if USING_VAR_NPZ:
            Z = var_kde_by_chr[my_chr]
        else:
            Z = np.zeros((KDE_NUMPOINTS_VAF, int(CONTIG_SIZES[my_chr]/WINDOW_SIZE)+1), dtype='float')
            if my_chr in var_het_by_chr:
                for vi in range(var_het_by_chr[my_chr].shape[1]):
                    my_vpos = int(var_het_by_chr[my_chr][0,vi]/WINDOW_SIZE)
                    my_vvaf = int(var_het_by_chr[my_chr][1,vi]*KDE_NUMPOINTS_VAF)
                    my_std_pos = KDE_STD_POS/WINDOW_SIZE
                    my_pos_buff = int(my_std_pos*3) # go out 3 stds on either side
                    my_vaf_buff = int(KDE_STD_VAF*3)
                    for zy in range(max(0,my_vpos-my_pos_buff), min(Z.shape[1],my_vpos+my_pos_buff)):
                        for zx in range(max(0,my_vvaf-my_vaf_buff), min(Z.shape[0],my_vvaf+my_vaf_buff)):
                            Z[zx,zy] += np.exp(log_px(zx, zy, my_vvaf, my_vpos, KDE_STD_VAF, my_std_pos))
                for zy in range(Z.shape[1]):
                    my_sum = np.sum(Z[:,zy])
                    if my_sum > 0.0:
                        Z[:,zy] /= my_sum
            var_kde_by_chr[my_chr] = np.array(Z, copy=True)
        #
        X, Y = np.meshgrid(range(0,len(Z[0])+1), range(0,len(Z)+1))
        mpl.pcolormesh(X,Y,Z)
        mpl.axis([0,len(Z[0]),0,len(Z)])
        mpl.yticks(KDE_YT, KDE_YL)
        mpl.ylabel('BAF')
        for tick in ax2.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        #
        ax3 = mpl.subplot(gs[2])
        if my_chr in cyto_by_chr:
            polygons = []
            p_color  = []
            p_alpha  = []
            for cdat in cyto_by_chr[my_chr]:
                pq = cdat[2][0]
                my_type = cdat[3]
                xp = [cdat[0], cdat[1]]
                yp = [-1, 1]
                if my_type == 'acen':
                    if pq == 'p':
                        polygons.append(Polygon(np.array([[xp[0],yp[0]], [xp[0],yp[1]], [xp[1],0]]), closed=True))
                    else:
                        polygons.append(Polygon(np.array([[xp[0],0], [xp[1],yp[1]], [xp[1],yp[0]]]), closed=True))
                else:
                    polygons.append(Polygon(np.array([[xp[0],yp[0]], [xp[0],yp[1]], [xp[1],yp[1]], [xp[1],yp[0]]]), closed=True))
                p_color.append(CYTOBAND_COLORS[my_type])
                p_alpha.append(0.8)
            for j in range(len(polygons)):
                ax3.add_collection(PatchCollection([polygons[j]], color=p_color[j], alpha=p_alpha[j], linewidth=0))
        mpl.xticks(xt,xl,rotation=70)
        mpl.yticks([],[])
        mpl.xlim(0,CONTIG_SIZES[my_chr])
        mpl.ylim(-1,1)
        mpl.ylabel(my_chr)
        mpl.tight_layout()
        mpl.savefig(f'{PLOT_DIR}cov_{my_chr}.png')
        mpl.close(fig)

        #
        # EXPERIMENTAL FEATURE: bimodal gaussian fit --> CNV calling
        #
        if REPORT_COPYNUM:
            sorted_het_coords = [int(n) for n in var_het_by_chr[my_chr][0,:]]
            sorted_het_vafs = [float(n) for n in var_het_by_chr[my_chr][1,:]]
            sorted_hom_coords = [int(n) for n in var_hom_by_chr[my_chr][0,:]]
            sorted_hom_vafs = [float(n) for n in var_hom_by_chr[my_chr][1,:]]
            cnv_bed_dat = []
            for vi in range(0, CONTIG_SIZES[my_chr], CNV_NUM_INDS * WINDOW_SIZE):
                start_coords = vi
                end_coords = vi + CNV_NUM_INDS * WINDOW_SIZE
                #
                my_cov_vector = plotted_cx_cy[my_chr][1][start_coords//WINDOW_SIZE:end_coords//WINDOW_SIZE]
                my_avg_cov = np.median(my_cov_vector)
                #
                (v_lb, v_ub) = find_indices_in_range(sorted_het_coords, start_coords, end_coords)
                my_window_hets = sorted_het_vafs[v_lb:v_ub]
                (v_lb_hom, v_ub_hom) = find_indices_in_range(sorted_hom_coords, start_coords, end_coords)
                my_window_homs = sorted_hom_vafs[v_lb_hom:v_ub_hom]
                if len(my_window_hets) >= CNV_MINVAR:
                    my_hethom_ratio = None
                    if len(my_window_homs):
                        my_hethom_ratio = len(my_window_hets) / len(my_window_homs)
                    bimodal_fit = fit_bimodal_gaussian(np.array(my_window_hets))
                    #print(bimodal_fit)
                    norm_ll_component_ratio = (bimodal_fit['component1_log_likelihood'] - bimodal_fit['component2_log_likelihood']) / len(my_window_hets)
                    print(my_chr, start_coords, end_coords, len(my_window_hets), f'{my_hethom_ratio:.3f}', f'{norm_ll_component_ratio:.3f}', f'{my_avg_cov:.3f}')
                    is_bimodal = bimodal_fit['single_gaussian_p-value'] < 0.05 and abs(norm_ll_component_ratio) < 10.0
                    # bimodal
                    if is_bimodal:
                        #plot_fn = f'{PLOT_DIR}bimodal_{my_chr}_{start_coords}_{end_coords}.png'
                        #plot_title = f'{my_chr}:{start_coords}-{end_coords}'
                        #plot_bimodal_fit(np.array(my_window_hets), bimodal_fit['A'], bimodal_fit['B'], plot_fn, plot_title=plot_title)
                        my_mean1 = 0.5 - bimodal_fit['A']
                        my_mean2 = 0.5 + bimodal_fit['A']
                        my_var = bimodal_fit['B']
                        my_norm_ll = bimodal_fit['max_log_likelihood'] / len(my_window_hets)
                        print(f'-- Bimodal: u1 = {my_mean1:.3f}, u2 = {my_mean2:.3f}, o^2 = {my_var:.3f}, nll = {my_norm_ll:.3f}')
                    # unimodal
                    else:
                        my_mean = np.mean(my_window_hets)
                        my_var = np.var(my_window_hets)
                        my_norm_ll = bimodal_fit['single_gaussian_log_likelihood'] / len(my_window_hets)
                        # possibly many outliers, lets use median instead of mean
                        if abs(norm_ll_component_ratio) > 10.0:
                            my_mean = np.median(my_window_hets)
                        print(f'-- Unimodal: u = {my_mean:.3f}, o^2 = {my_var:.3f}, nll = {my_norm_ll:.3f}')
                    # example of appending to cnv_bed_dat:
                    # cnv_bed_dat.append((my_chr, start_coords, end_coords, cnv_assignment, cnv_likelihood, avg_coverage))
            exit(1)
            #
            # merge windows into larger CNV calls
            #
            if cnv_bed_dat:
                cnv_windows = [[0, 1, cnv_bed_dat[0][3]]]
                current_copynum = cnv_bed_dat[0][3]
                for i,cbd in enumerate(cnv_bed_dat):
                    if i == 0:
                        continue
                    current_copynum = cnv_bed_dat[i][3]
                    if current_copynum == cnv_windows[-1][2]:
                        cnv_windows[-1][1] = i+1
                    else:
                        cnv_windows.append([i, i+1, current_copynum])
                for cw in cnv_windows:
                    avg_cnv_likelihood = np.mean([cnv_bed_dat[n][4] for n in range(cw[0], cw[1])])
                    avg_cnv_coverage = np.mean([cnv_bed_dat[n][5] for n in range(cw[0], cw[1])])
                    out_cnv_assignment = cw[2]
                    out_cnv_start = cnv_bed_dat[cw[0]][1]
                    out_cnv_end = cnv_bed_dat[cw[1]-1][2]
                    cnv_bed_out.append((my_chr, out_cnv_start, out_cnv_end, out_cnv_assignment, avg_cnv_likelihood, avg_cnv_coverage))

        sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
        sys.stdout.flush()

    #
    # save parsed vcf data
    #
    if USING_VAR_NPZ is False:
        var_npz_outdict = {}
        var_npz_outdict['extra_covwin'] = WINDOW_SIZE
        var_npz_outdict['extra_varwin'] = VAR_WINDOW
        #
        for my_chr in var_kde_by_chr.keys():
            var_npz_outdict[f'kde_{my_chr}'] = var_kde_by_chr[my_chr]
        for my_chr in var_het_by_chr.keys():
            var_npz_outdict[f'het_{my_chr}'] = var_het_by_chr[my_chr]
        for my_chr in var_hom_by_chr.keys():
            var_npz_outdict[f'hom_{my_chr}'] = var_hom_by_chr[my_chr]
        # save it all together in a single npz
        np.savez_compressed(VAF_NPZ, var_filt_whitelist=','.join(VAR_FILT_WHITELIST), var_filt_blacklist=','.join(VAR_FILT_BLACKLIST), **var_npz_outdict)

    #
    # whole genome plot (concatenated coverage)
    #
    plot_fn = f'{PLOT_DIR}cov_wholegenome.png'
    if SAMP_NAME:
        plot_fn = f'{PLOT_DIR}cov_wholegenome_{SAMP_NAME}.png'
    fig = mpl.figure(1, figsize=(30,10), dpi=200)
    ax1 = mpl.subplot(311)
    current_x_offset = 0
    current_color_ind = 0
    concat_var_matrix = None
    concat_het_dens = None
    concat_hom_dens = None
    chrom_xticks_major = [0]
    chrom_xlabels_major = ['']
    chrom_xticks_minor = []
    chrom_xlabels_minor = []
    for my_chr in sorted_chr:
        my_color = CHROM_COLOR_CYCLE[current_color_ind % len(CHROM_COLOR_CYCLE)]
        (cx, cy) = plotted_cx_cy[my_chr]
        if my_chr in var_kde_by_chr:
            Zvar = var_kde_by_chr[my_chr]
        else:
            Zvar = np.zeros((KDE_NUMPOINTS_VAF, int(CONTIG_SIZES[my_chr]/WINDOW_SIZE)+1), dtype='float')
        if my_chr == sorted_chr[0]:
            concat_var_matrix = Zvar
            concat_het_dens = het_dens_by_chr[my_chr]
            concat_hom_dens = hom_dens_by_chr[my_chr]
        else:
            concat_var_matrix = np.concatenate((concat_var_matrix, Zvar), axis=1)
            concat_het_dens = np.concatenate((concat_het_dens, het_dens_by_chr[my_chr]), axis=0)
            concat_hom_dens = np.concatenate((concat_hom_dens, hom_dens_by_chr[my_chr]), axis=0)
        #
        if my_chr in unstable_by_chr:
            for ur in unstable_by_chr[my_chr]:
                w1 = max(math.floor(ur[0]/WINDOW_SIZE), 0)
                w2 = min(math.ceil(ur[1]/WINDOW_SIZE), len(cy)-1)
                cy[w1:w2+1] = COV_YT[0] - 1.0
        #
        mpl.scatter(cx + current_x_offset, cy, s=0.5, color=my_color)
        chrom_xticks_minor.append(current_x_offset + 0.5 * len(cx) * WINDOW_SIZE)
        chrom_xlabels_minor.append(my_chr)
        chrom_xticks_major.append(current_x_offset + len(cx) * WINDOW_SIZE)
        chrom_xlabels_major.append('')
        current_x_offset += len(cx) * WINDOW_SIZE
        current_color_ind += 1
    mpl.xticks(chrom_xticks_major, chrom_xlabels_major)
    mpl.xticks(chrom_xticks_minor, chrom_xlabels_minor, minor=True)
    mpl.xlim([0, current_x_offset])
    mpl.ylim(COV_YT[0], COV_YT[-1])
    mpl.grid(which='major', linestyle='--', alpha=0.6)
    mpl.ylabel('log2 cov change')
    mpl.title(f'average coverage: {all_avg_cov[0]:0.3f}')
    for tick in ax1.xaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    #
    ax2 = mpl.subplot(312)
    Z = concat_var_matrix
    X, Y = np.meshgrid(range(0,len(Z[0])+1), range(0,len(Z)+1))
    mpl.pcolormesh(X,Y,Z)
    mpl.axis([0,len(Z[0]),0,len(Z)])
    mpl.yticks(KDE_YT, KDE_YL)
    mpl.ylabel('BAF')
    for tick in ax2.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    #
    ax3 = mpl.subplot(313)
    mpl.plot([VAR_WINDOW*n for n in range(len(concat_het_dens))], concat_het_dens, color='blue', alpha=0.5)
    mpl.plot([VAR_WINDOW*n for n in range(len(concat_hom_dens))], concat_hom_dens, color='red', alpha=0.5)
    mpl.xlim(0,VAR_WINDOW*len(concat_het_dens))
    mpl.ylim(bottom=0)
    mpl.ylabel('variant density')
    for tick in ax3.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    #
    mpl.tight_layout()
    mpl.savefig(plot_fn)
    mpl.close(fig)

    #
    # whole genome plot (stacked annotations)
    #
    pass

    if cnv_bed_out:
        with open(CNV_BED, 'w') as f:
            for n in cnv_bed_out:
                if n[3] != 2: # don't report diploid
                    f.write(f'{n[0]}\t{n[1]}\t{n[2]}\t{n[3]}\t{n[4]:.2f}\n')

    print(f'average coverage: {all_avg_cov[0]:0.3f}')
    if len(all_bed_result):
        print('region coverage:')
        for n in all_bed_result:
            print(f' - {n[0][2]}: {n[1]:0.3f}')
    with open(f'{OUT_DIR}region_coverage.tsv','w') as f:
        f.write('region\tmean_cov\tmedian_cov\tstd\n')
        f.write(f'whole_genome\t{all_avg_cov[0]:0.3f}\t{all_avg_cov[1]:0.3f}\t{all_avg_cov[2]:0.3f}\n')
        for n in all_bed_result:
            f.write(f'{n[0][2]}\t{n[1]:0.3f}\t{n[2]:0.3f}\t{n[3]:0.3f}\n')


if __name__ == '__main__':
    main()
