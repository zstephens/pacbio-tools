#!/usr/bin/env python
import argparse
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


def get_template_vaf(vaf_list, WINDOW_SIZE, vaf_std):
    template = np.zeros((KDE_NUMPOINTS_VAF), dtype='float')
    my_std_pos = KDE_STD_POS/WINDOW_SIZE
    for my_vvaf in vaf_list:
        for zx in range(template.shape[0]):
            template[zx] += np.exp(log_px(zx, 0, int(my_vvaf*KDE_NUMPOINTS_VAF), 0, vaf_std, my_std_pos))
    template /= np.sum(template)
    return template


def emd(pdf_a, pdf_b):
    return wasserstein_distance(np.arange(len(pdf_a)), np.arange(len(pdf_b)), pdf_a, pdf_b)


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='plot_coverage.py')
    parser.add_argument('-i',  type=str, required=True,  metavar='<str>', help="* input.bam")
    parser.add_argument('-o',  type=str, required=True,  metavar='<str>', help="* output_dir/")
    parser.add_argument('-r',  type=str, required=False, metavar='<str>', help="refname: t2t / hg38 / hg19",   default='t2t')
    parser.add_argument('-q',  type=int, required=False, metavar='<int>', help="minimum MAPQ",                 default=0)
    parser.add_argument('-w',  type=int, required=False, metavar='<int>', help="window size for downsampling", default=10000)
    parser.add_argument('-b',  type=str, required=False, metavar='<str>', help="bed of regions to query",      default='')
    parser.add_argument('-v',  type=str, required=False, metavar='<str>', help="input.vcf (somatic)",          default='')
    parser.add_argument('-s',  type=str, required=False, metavar='<str>', help="sample name",                  default='')
    parser.add_argument('-rt', type=str, required=False, metavar='<str>', help="read type: CCS / CLR / ONT",   default='CCS')
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

    MIN_MAPQ = max(0,args.q)
    WINDOW_SIZE = max(1,args.w)

    REF_VERS  = args.r
    BED_FILE  = args.b
    IN_VCF    = args.v
    SAMP_NAME = args.s

    REPORT_COPYNUM = False

    OUT_NPZ = f'{OUT_DIR}cov.npz'
    VAF_NPZ = f'{OUT_DIR}vaf.npz'
    if SAMP_NAME:
        OUT_NPZ = f'{OUT_DIR}cov_{SAMP_NAME}.npz'
        VAF_NPZ = f'{OUT_DIR}vaf_{SAMP_NAME}.npz'

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
    in_variants = {}
    var_kde_by_chr = {}
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
                    my_filt = splt[6]
                    if my_filt in ['PASS', 'NonSomatic']:
                        fmt_split = splt[8].split(':')
                        dat_split = splt[9].split(':')
                        if 'GT' in fmt_split and 'AF' in fmt_split:
                            ind_gt = fmt_split.index('GT')
                            ind_af = fmt_split.index('AF')
                            my_gt = dat_split[ind_gt]
                            my_af = float(dat_split[ind_af])
                            if my_gt == '1/1' or my_gt == '1|1' or my_af >= 0.950:
                                pass
                            else:
                                if my_chr not in in_variants:
                                    in_variants[my_chr] = [[], []]
                                in_variants[my_chr][0].append(my_pos)
                                in_variants[my_chr][1].append(my_af)
                                #print(splt[0], splt[1], splt[3], splt[4], my_filt, my_gt, my_af)
            f.close()
            sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
            sys.stdout.flush()
        #
        elif IN_VCF[-4:].lower() == '.npz':
            print('reading from an existing npz archive instead of vcf...')
            in_npz = np.load(IN_VCF)
            var_kde_by_chr = {k:in_npz[k] for k in sorted_chr}
            USING_VAR_NPZ = True
        #
        else:
            print('Error: -v must be .vcf or .vcf.gz or .npz')
            exit(1)

    #
    # make VAF templates
    #
    VAF_TEMPLATES = {}
    VAF_TEMPLATES['hom'] = get_template_vaf([0.100], WINDOW_SIZE, KDE_STD_VAF*1.0)
    VAF_TEMPLATES['het'] = get_template_vaf([0.500], WINDOW_SIZE, KDE_STD_VAF*2.0)
    for copynum in range(3,6):
        for numer in range(1,int(copynum/2)+1):
            if numer*2 != copynum:
                my_vaf = numer/copynum
                VAF_TEMPLATES[f'{copynum}-{numer}'] = get_template_vaf([my_vaf, 1.0-my_vaf], WINDOW_SIZE, KDE_STD_VAF*1.0)

    #
    # determine average coverage across whole genome
    # -- in most cases this will correspond to 2 copies, but not always
    #
    all_win = []
    for my_chr in sorted_chr:
        if my_chr in UNSTABLE_CHR:
            continue
        cy = np.copy(covdat_by_ref[my_chr])
        if my_chr in unstable_by_chr:
            for ur in unstable_by_chr[my_chr]:
                w1 = max(math.floor(ur[0]/WINDOW_SIZE), 0)
                w2 = min(math.ceil(ur[1]/WINDOW_SIZE), len(cy)-1)
                cy[w1:w2+1] = -1.0
        all_win.extend(cy[cy >= 0.0].tolist())
    all_avg_cov = (np.mean(all_win), np.median(all_win), np.std(all_win))
    avg_log2 = np.log2(np.median(all_win))
    del all_win

    plotted_cx_cy = {}

    #
    # plotting!
    #
    fig_width_scalar = 11.5/CONTIG_SIZES['chr1']
    fig_width_buffer = 0.5
    fig_width_min    = 2.0
    fig_height       = 5.0
    for my_chr in sorted_chr:
        sys.stdout.write(f'making plots for {my_chr}...')
        sys.stdout.flush()
        tt = time.perf_counter()
        #
        xt = np.arange(0,CONTIG_SIZES[my_chr],10000000)
        xl = [f'{n*10}M' for n in range(len(xt))]
        with np.errstate(divide='ignore'):
            cy = np.log2(covdat_by_ref[my_chr]) - avg_log2
        cx = np.array([n*WINDOW_SIZE + WINDOW_SIZE/2 for n in range(len(cy))])
        plotted_cx_cy[my_chr] = (np.copy(cx), np.copy(cy))
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
            if my_chr in in_variants:
                #(markerline, stemlines, baseline) = mpl.stem(in_variants[my_chr][0], in_variants[my_chr][1])
                #mpl.setp(baseline, visible=False)
                #mpl.setp(stemlines, visible=False)
                #mpl.setp(markerline, markersize=1)
                for vi in range(len(in_variants[my_chr][0])):
                    #print(vi, '/', len(in_variants[my_chr][0]))
                    my_vpos = int(in_variants[my_chr][0][vi]/WINDOW_SIZE)
                    my_vvaf = int(in_variants[my_chr][1][vi]*KDE_NUMPOINTS_VAF)
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
            Z = np.array(Z[::-1])
            var_kde_by_chr[my_chr] = np.array(Z, copy=True)
        #
        # EXPERIMENTAL FEATURE: VAF template detection
        # -- goes cytoband by cytoband --> sliding windows within each cytoband
        #
        if REPORT_COPYNUM:
            if my_chr not in UNSTABLE_CHR:
                for cdat in cyto_by_chr[my_chr]:
                    my_type = cdat[3]
                    if my_type not in UNSTABLE_REGION:
                        xp = [int(cdat[0]/WINDOW_SIZE), int(cdat[1]/WINDOW_SIZE)+1]
                        if np.sum(Z[:,xp[0]:xp[1]]) <= 0.0:
                            continue
                        avg_signal = np.mean(Z[:,xp[0]:xp[1]], axis=1)
                        my_sum = np.sum(avg_signal)
                        if my_sum > 0.0:
                            avg_signal /= my_sum
                            sorted_scores = []
                            for template_name,vaf_template in VAF_TEMPLATES.items():
                                my_dist = emd(vaf_template, avg_signal)
                                sorted_scores.append((my_dist, template_name))
                                ####fig999 = mpl.figure(999)
                                ####mpl.plot(np.arange(len(vaf_template)), vaf_template, '--r')
                                ####mpl.plot(np.arange(len(avg_signal)), avg_signal, '-b')
                                ####mpl.title(f'{my_chr}:{my_type}:{xp[0]}-{xp[1]} vs. {template_name}')
                                ####mpl.legend([f'{my_dist:0.3f}'])
                                ####mpl.savefig(f'{PLOT_DIR}{my_chr}_{my_type}_{xp[0]}-{xp[1]}_{template_name}.png')
                                ####mpl.close(fig999)
                            sorted_scores = sorted(sorted_scores)
                            print(sorted_scores[:3])
                #Z = np.tile(vaf_template, (1,Z.shape[1]))
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
        mpl.tight_layout()
        mpl.savefig(f'{PLOT_DIR}cov_{my_chr}.png')
        mpl.close(fig)
        #
        sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
        sys.stdout.flush()

    if USING_VAR_NPZ is False:
        np.savez_compressed(VAF_NPZ, **var_kde_by_chr)

    #
    # whole genome plot
    #
    plot_fn = f'{PLOT_DIR}cov_wholegenome.png'
    if SAMP_NAME:
        plot_fn = f'{PLOT_DIR}cov_wholegenome_{SAMP_NAME}.png'
    fig = mpl.figure(1, figsize=(30,10), dpi=200)
    ax1 = mpl.subplot(211)
    current_x_offset = 0
    current_color_ind = 0
    concat_var_matrix = None
    chrom_xticks_major = [0]
    chrom_xlabels_major = ['']
    chrom_xticks_minor = []
    chrom_xlabels_minor = []
    for my_chr in sorted_chr:
        my_color = CHROM_COLOR_CYCLE[current_color_ind % len(CHROM_COLOR_CYCLE)]
        (cx, cy) = plotted_cx_cy[my_chr]
        Zvar = var_kde_by_chr[my_chr]
        if my_chr == sorted_chr[0]:
            concat_var_matrix = Zvar
        else:
            concat_var_matrix = np.concatenate((concat_var_matrix, Zvar), axis=1)
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
    ax2 = mpl.subplot(212)
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
    mpl.tight_layout()
    mpl.savefig(plot_fn)
    mpl.close(fig)

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
