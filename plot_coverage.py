#!/usr/bin/env python
import argparse
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

# mask these regions when computing average coverage
UNSTABLE_REGION = ['acen', 'gvar', 'stalk']
UNSTABLE_CHR = ['chrM']


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


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='plot_coverage.py')
    parser.add_argument('-i',  type=str, required=True,  metavar='<str>', help="* input.bam")
    parser.add_argument('-o',  type=str, required=True,  metavar='<str>', help="* output_dir/")
    parser.add_argument('-r',  type=str, required=False, metavar='<str>', help="refname: t2t / hg38 / hg19",   default='t2t')
    parser.add_argument('-q',  type=int, required=False, metavar='<int>', help="minimum MAPQ",                 default=0)
    parser.add_argument('-w',  type=int, required=False, metavar='<int>', help="window size for downsampling", default=10000)
    parser.add_argument('-b',  type=str, required=False, metavar='<str>', help="bed of regions to query",      default='')
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
    OUT_NPZ = OUT_DIR+'cov.npz'

    MIN_MAPQ = max(0,args.q)
    WINDOW_SIZE = max(1,args.w)

    REF_VERS = args.r
    BED_FILE = args.b

    bed_regions = {}
    if len(BED_FILE):
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
    #
    else:
        print('Error: -i must be .bam or .npz')
        exit(1)

    #
    sys.stdout.write('making plots...')
    sys.stdout.flush()
    tt = time.perf_counter()
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
    avg_log2 = np.log2(np.mean(all_win))
    del all_win
    #
    fig_width_scalar = 11.5/CONTIG_SIZES['chr1']
    fig_width_buffer = 0.5
    fig_width_min    = 2.0
    fig_height       = 4.0
    for my_chr in sorted_chr:
        xt = np.arange(0,CONTIG_SIZES[my_chr],10000000)
        xl = [f'{n*10}M' for n in range(len(xt))]
        with np.errstate(divide='ignore'):
            cy = np.log2(covdat_by_ref[my_chr]) - avg_log2
        cx = np.array([n*WINDOW_SIZE + WINDOW_SIZE/2 for n in range(len(cy))])
        #
        my_width = max(CONTIG_SIZES[my_chr]*fig_width_scalar + fig_width_buffer, fig_width_min)
        fig = mpl.figure(1, figsize=(my_width,fig_height), dpi=200)
        gs = gridspec.GridSpec(2, 1, height_ratios=[8,1])
        ax1 = mpl.subplot(gs[0])
        mpl.scatter(cx, cy, s=1, c='black')
        mpl.xlim(0,CONTIG_SIZES[my_chr])
        mpl.ylim(-3,3)
        mpl.xticks(xt,xl)
        mpl.grid(which='both', linestyle='--', alpha=0.6)
        for tick in ax1.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        #
        ax2 = mpl.subplot(gs[1])
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
                ax2.add_collection(PatchCollection([polygons[j]], color=p_color[j], alpha=p_alpha[j], linewidth=0))
        mpl.xticks(xt,xl,rotation=70)
        mpl.yticks([],[])
        mpl.xlim(0,CONTIG_SIZES[my_chr])
        mpl.ylim(-1,1)
        #
        mpl.tight_layout()
        mpl.savefig(f'{OUT_DIR}cov_{my_chr}.png')
        mpl.close(fig)
    sys.stdout.write(f' ({int(time.perf_counter() - tt)} sec)\n')
    sys.stdout.flush()
    #
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
