import os
import sys
import re
import gzip
import json
import argparse

import numpy as np

import matplotlib.pyplot as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def getColor(i,N,colormap='jet'):
	cm = mpl.get_cmap(colormap) 
	cNorm  = colors.Normalize(vmin=0, vmax=N+1)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
	colorVal = scalarMap.to_rgba(i)
	return colorVal

HUMAN_CHR  = [str(n) for n in range(1,22+1)] + ['X', 'Y']
HUMAN_CHR += ['chr'+n for n in HUMAN_CHR]
HUMAN_CHR  = {n:True for n in HUMAN_CHR}

TEL_REF_DICT = {'telomere_chr1_p_0_3000'               : 'tel1p',
                'telomere_chr1_q_248384000_248387328'  : 'tel1q',
                'telomere_chr2_p_0_3600'               : 'tel2p',
                'telomere_chr2_q_242693800_242696752'  : 'tel2q',
                'telomere_chr3_p_0_2800'               : 'tel3p',
                'telomere_chr3_p_200904400_200905400'  : 'tel3q?',	# weird extra telomere repeats
                'telomere_chr3_q_201101200_201105948'  : 'tel3q',
                'telomere_chr4_p_0_3400'               : 'tel4p',
                'telomere_chr4_q_193572600_193574945'  : 'tel4q',
                'telomere_chr5_p_0_2400'               : 'tel5p',
                'telomere_chr5_q_182043800_182045439'  : 'tel5q',
                'telomere_chr6_p_0_3000'               : 'tel6p',
                'telomere_chr6_q_172123600_172126628'  : 'tel6q',
                'telomere_chr7_p_0_3600'               : 'tel7p',
                'telomere_chr7_q_160565200_160567428'  : 'tel7q',
                'telomere_chr8_p_0_2400'               : 'tel8p',
                'telomere_chr8_q_146257200_146259331'  : 'tel8q',
                'telomere_chr9_p_0_3800'               : 'tel9p',
                'telomere_chr9_q_150614400_150617247'  : 'tel9q',
                'telomere_chr10_p_0_2800'              : 'tel10p',
                'telomere_chr10_q_134754800_134758134' : 'tel10q',
                'telomere_chr11_p_0_2200'              : 'tel11p',
                'telomere_chr11_p_201200_202800'       : 'tel11p?',	# weird extra telomere repeats
                'telomere_chr11_q_135125000_135127769' : 'tel11q',
                'telomere_chr12_p_0_3200'              : 'tel12p',
                'telomere_chr12_q_133322000_133324548' : 'tel12q',
                'telomere_chr13_p_0_3000'              : 'tel13p',
                'telomere_chr13_q_113563000_113566686' : 'tel13q',
                'telomere_chr14_p_0_2400'              : 'tel14p',
                'telomere_chr14_q_101159600_101161492' : 'tel14q',
                'telomere_chr15_p_0_3600'              : 'tel15p',
                'telomere_chr15_q_99750200_99753195'   : 'tel15q',
                'telomere_chr16_p_0_2600'              : 'tel16p',
                'telomere_chr16_q_96327400_96330374'   : 'tel16q',
                'telomere_chr17_p_0_2200'              : 'tel17p',
                'telomere_chr17_q_84273800_84276897'   : 'tel17q',
                'telomere_chr18_p_0_2400'              : 'tel18p',
                'telomere_chr18_q_80539000_80542538'   : 'tel18q',
                'telomere_chr19_p_0_2400'              : 'tel19p',
                'telomere_chr19_q_61704200_61707364'   : 'tel19q',
                'telomere_chr20_p_0_2800'              : 'tel20p',
                'telomere_chr20_q_66207000_66210255'   : 'tel20q',
                'telomere_chr21_p_0_3000'              : 'tel21p',
                'telomere_chr21_q_45086000_45090682'   : 'tel21q',
                'telomere_chr22_p_0_4600'              : 'tel22p',
                'telomere_chr22_q_51321800_51324926'   : 'tel22q',
                'telomere_chrX_p_0_2000'               : 'telXp',
                'telomere_chrX_q_154256800_154259566'  : 'telXq',
                'telomere_TAACCC_ad-infinitum'         : 'telTAACCC'}

TEL_BOUNDARY = {}
for k in TEL_REF_DICT.keys():
	splt = k.split('_')
	if splt[1] in HUMAN_CHR:
		if splt[1] not in TEL_BOUNDARY:
			TEL_BOUNDARY[splt[1]] = [999999999999, 0]
		if splt[2] == 'p':
			TEL_BOUNDARY[splt[1]][0] = min([TEL_BOUNDARY[splt[1]][0], int(splt[4])])
		elif splt[2] == 'q':
			TEL_BOUNDARY[splt[1]][1] = max([TEL_BOUNDARY[splt[1]][1], int(splt[3])])
#for k in sorted(TEL_BOUNDARY.keys()):
#	print(k, TEL_BOUNDARY[k])
#exit(1)

#
#	returns ( min_dist_to_telomere_boundary, 'p'/'q' )
#
SUBTEL_BUFF         = 100000
SUBTEL_BUFF_RELAXED = 500000	# used for trying to salvage tels not anchored within SUBTEL_BUFF
def is_subtel(my_chr, my_pos_list, relaxed=False):
	if my_chr in TEL_BOUNDARY:
		p_dist = min([abs(n - TEL_BOUNDARY[my_chr][0]) for n in my_pos_list])
		q_dist = min([abs(n - TEL_BOUNDARY[my_chr][1]) for n in my_pos_list])
		thresh = SUBTEL_BUFF
		if relaxed:
			thresh = SUBTEL_BUFF_RELAXED
		if min([p_dist, q_dist]) <= thresh:
			if p_dist < q_dist:
				return (p_dist, 'p')
			else:
				return (q_dist, 'q')
	return None

# returns reference span
REF_CHAR  = 'MX=D'
READ_CHAR = 'MX=I'
CLIP_CHAR = 'SH'
def parse_cigar(cigar):
	letters = re.split(r"\d+",cigar)[1:]
	numbers = [int(n) for n in re.findall(r"\d+",cigar)]
	startPos = 0
	if letters[0] in CLIP_CHAR:
		startPos = numbers[0]
	endClip = 0
	if len(letters) > 1 and letters[-1] in CLIP_CHAR:
		endClip = numbers[-1]
	adj  = 0
	radj = 0
	for i in range(len(letters)):
		if letters[i] in REF_CHAR:
			adj += numbers[i]
		if letters[i] in READ_CHAR:
			radj += numbers[i]
	return (startPos, adj, radj, endClip)

def exists_and_is_nonZero(fn):
	if os.path.isfile(fn):
		if os.path.getsize(fn) > 0:
			return True
	return False

def makedir(d):
	if not os.path.isdir(d):
		os.system('mkdir '+d)

def get_nearest_transcript(myChr, pos, bedDat, max_dist=20000):
	if myChr in bedDat:
		# lazy and slow, but it gets the job done!
		closest_dist = 99999999999
		closest_meta = ''
		for n in bedDat[myChr]:
			if pos >= n[0] and pos <= n[1]:
				closest_dist = 0
				closest_meta = [n[2], n[3]]
				break
			my_dist = min([abs(pos-n[0]), abs(pos-n[1])])
			if my_dist < closest_dist:
				closest_dist = my_dist
				closest_meta = [n[2], n[3]]
		if closest_dist <= max_dist:
			return (closest_dist, closest_meta)
	return None

def is_same_type(t1, t2):
	if t1 == t2:
		return True
	if (t1 == 'tel' and t2 == 'tel?') or (t1 == 'tel?' and t2 == 'tel'):
		return True
	return False

RC_DICT = {'A':'T','C':'G','G':'C','T':'A','N':'N'}
def RC(s):
	return ''.join([RC_DICT[n] for n in s[::-1]])

TEL_REPEAT_SEQ  = ['TTAGGG', 'TCAGGG', 'TGAGGG', 'TTGGGG']
TEL_REPEAT_SEQ += [RC(n) for n in TEL_REPEAT_SEQ]

def count_tel_repeat_bases_in_string(s):
	t = np.zeros(len(s))
	for i in range(len(s)-6):
		for n in TEL_REPEAT_SEQ:
			if s[i:i+len(n)] == n:
				t[i:i+len(n)] = 1
	return int(np.sum(t))

CASE_NUMBER_COMPLEX = 97
CASE_NUMBER_TELONLY = 98
CASE_NUMBER_DISCARD = 99

# add the telomere content of reference alignments to their adjacent telomere alignments
# if at least this fraction of the alignment is comprised of telomere sequences
MIN_TEL_FRAC_TO_ADD_TELSEQ_BASES = 0.05
# only consider sub-seq-tel cases if len(sub) is at least this much larger than len(seq)
MIN_SUB_TO_SEQ_LEN_RATIO = 2.0

#
#
#
parser = argparse.ArgumentParser(description='plot_telomere_long_reads.py')
parser.add_argument('-i', type=str, required=True, metavar='<str>', help="* input.sam")
parser.add_argument('-o', type=str, required=True, metavar='<str>', help="* output_dir/")
parser.add_argument('-c', type=str, required=False, default=None,   metavar='<str>',     help="tsv of ground truth tel lens to compare against")
parser.add_argument('-j', type=str, required=False, default=None,   metavar='<str>',     help="skip read processing and just plot tel lens from this json")
parser.add_argument('--skip-plot',  required=False, default=False,  action='store_true', help='skip individual read plotting')
parser.add_argument('--pbsim',      required=False, default=False,  action='store_true', help='input is pbsim reads, print out confusion matrix')
args = parser.parse_args()

INPUT_SAM  = args.i
OUT_DIR    = args.o
if OUT_DIR[-1] != '/':
	OUT_DIR += '/'
makedir(OUT_DIR)
OUT_REPORT = OUT_DIR + 'junctions.tsv'
TEL_JSON   = OUT_DIR + 'tel_lens.json'
TEL_FASTA  = OUT_DIR + 'tel_sequences.fa'
PLOT_DIR   = OUT_DIR + 'read_plots/'
QCPLOT_DIR = OUT_DIR + 'qc/'
makedir(PLOT_DIR)
makedir(QCPLOT_DIR)

SUMMARY_SCATTER = OUT_DIR + 'tel_lens_scatter.png'
SUMMARY_VIOLIN  = OUT_DIR + 'tel_lens_violin.png'
CONFUSION_PLOT  = OUT_DIR + 'subtel_confusion_matrix.png'

INPUT_JSON = args.j
USE_JSON   = False
ANCHORED_TEL_JSON = None
if INPUT_JSON != None and exists_and_is_nonZero(INPUT_JSON):
	print('== Ignoring SAM input, plotting tel lens from json instead ==')
	f = open(INPUT_JSON, 'r')
	ANCHORED_TEL_JSON = json.load(f)
	f.close()
	USE_JSON = True

INPUT_COMP = args.c
COMP_DICT  = {}
if INPUT_COMP != None and exists_and_is_nonZero(INPUT_COMP):
	print('== Reading in ground truth tel lens for comparison ==')
	f = open(INPUT_COMP, 'r')
	for line in f:
		splt = line.strip().split('\t')
		COMP_DICT[splt[0]] = int(splt[1])
	f.close()

SKIP_READ_PLOTS = args.skip_plot
READS_ARE_PBSIM = args.pbsim

mpl.rcParams.update({'font.size': 18, 'font.weight':'bold'})

if ANCHORED_TEL_JSON == None:
	# [readpos_start, readpos_end, ref, pos_start, pos_end, orientation, mapq]
	ALIGNMENTS_BY_RNAME = {}
	READDAT_BY_RNAME = {}
	READLEN_BY_RNAME = {}

	print('reading SAM input...')
	f = open(INPUT_SAM, 'r')
	for line in f:
		splt = line.strip().split('\t')
		cigar = splt[5]
		flag  = int(splt[1])
		ref   = splt[2]
		pos   = int(splt[3])
		rdat  = splt[9]
		rnm   = splt[0]
		mapq  = int(splt[4])

		if ref in TEL_REF_DICT:
			ref = TEL_REF_DICT[ref]

		orientation = 'FWD'
		if flag&16:
			orientation = 'REV'

		cigDat = parse_cigar(cigar)

		readPos1 = cigDat[0]
		readPos2 = cigDat[0] + cigDat[2]
		readLen  = cigDat[0] + cigDat[2] + cigDat[3]
		if orientation == 'REV':
			[readPos1, readPos2] = [readLen - readPos2, readLen - readPos1]

		if rnm not in ALIGNMENTS_BY_RNAME:
			ALIGNMENTS_BY_RNAME[rnm] = []
			READLEN_BY_RNAME[rnm]    = 0
		if orientation == 'FWD':
			pos1, pos2 = pos, pos + cigDat[1]
		elif orientation == 'REV':
			pos1, pos2 = pos + cigDat[1], pos
		ALIGNMENTS_BY_RNAME[rnm].append([readPos1, readPos2, ref, pos1, pos2, orientation, mapq])

		READDAT_BY_RNAME[rnm] = rdat
		READLEN_BY_RNAME[rnm] = max([READLEN_BY_RNAME[rnm], readLen])
	f.close()

	# get rid of pathological cases where grep "telomere" found a single alignment where "telomere"
	# just so happened to be in the quality score string. WHAT ARE THE ODDS???
	for k in sorted(ALIGNMENTS_BY_RNAME.keys()):
		if len(ALIGNMENTS_BY_RNAME) <= 1:
			del ALIGNMENTS_BY_RNAME[k]

	#
	#
	#
	f_out  = open(OUT_REPORT, 'w')
	vseq_out = {}
	DELIM_MAJOR = '='
	DELIM_MINOR = '-'
	nPlot  = 1

	tel_count    = {}
	case_count   = {}
	skip_count   = {'only_tel':0, '2_tel_flank':0, '0_tel_flank':0}
	ALL_TEL_LENS = []
	ANCHORED_TEL = {}
	ANCHORED_POS = {}
	CONF_DAT     = {}
	CHR_FAILED_TO_ANCHOR = {}

	print('processing reads...')
	for k in sorted(ALIGNMENTS_BY_RNAME.keys()):
		abns_k = sorted(ALIGNMENTS_BY_RNAME[k])

		#print(k)
		#for n in abns_k:
		#	print(n)
		#continue

		for i in range(len(abns_k)):
			n = sorted(abns_k)[i]
			#print(n)
			if i >= 1:
				if n[2] in HUMAN_CHR and abns_k[i-1][2] not in HUMAN_CHR:
					qc1 = str(n[1]-n[0]) + ',' + str(n[6])
					qc2 = str(abns_k[i-1][1]-abns_k[i-1][0]) + ',' + str(abns_k[i-1][6])
					qc3 = str(n[0]-abns_k[i-1][1])
					outStr = k + '\t' + n[2] + ':' + str(n[3]) + '\t' + abns_k[i-1][2] + ':' + str(abns_k[i-1][4])
					f_out.write(outStr + '\t' + qc1 + '\t' + qc2 + '\t' + qc3 + '\n')
			if i < len(abns_k)-1:
				if n[2] in HUMAN_CHR and abns_k[i+1][2] not in HUMAN_CHR:
					qc1 = str(n[1]-n[0]) + ',' + str(n[6])
					qc2 = str(abns_k[i+1][1]-abns_k[i+1][0]) + ',' + str(abns_k[i+1][6])
					qc3 = str(abns_k[i+1][0]-n[1])
					outStr = k + '\t' + n[2] + ':' + str(n[4]) + '\t' + abns_k[i+1][2] + ':' + str(abns_k[i+1][3])
					f_out.write(outStr + '\t' + qc1 + '\t' + qc2 + '\t' + qc3 + '\n')
		#print('')
		
		viral_spans = []
		for i in range(len(abns_k)):
			n = abns_k[i]
			if n[2] not in HUMAN_CHR:
				if len(viral_spans) == 0:
					viral_spans.append([i])
				else:
					if i == viral_spans[-1][-1]+1:
						viral_spans[-1].append(i)
					else:
						viral_spans.append([i])

		#print(abns_k)
		#print(viral_spans)
		#print('')
		for n in viral_spans:

			(anchored_left, anchored_right)  = (False, False)
			(anchorseq_left, anchorseq_right) = ('', '')
			affected_genes = []
			if min(n) > 0:
				anchored_left = True
				# anchor seq format: "chr:pos:orr:gap:nearest_gene"
				a = abns_k[min(n)-1]
				out_nge = ''
				#nearest_hit = get_nearest_transcript(a[2], a[4], TRANSCRIPT_TRACK)
				nearest_hit = None
				if nearest_hit != None:
					out_nge = nearest_hit[1][1]
					affected_genes.append(out_nge)
				anchorseq_left = a[2] + ':' + str(a[4]) + ':' + a[5] + ':' + str(abns_k[min(n)][0] - a[1]) + ':' + out_nge
			if max(n) < len(abns_k)-1:
				anchored_right = True
				# anchor seq format: "chr:pos:orr:gap:nearest_gene"
				a = abns_k[max(n)+1]
				out_nge = ''
				#nearest_hit = get_nearest_transcript(a[2], a[3], TRANSCRIPT_TRACK)
				nearest_hit = None
				if nearest_hit != None:
					out_nge = nearest_hit[1][1]
					affected_genes.append(out_nge)
				anchorseq_right = a[2] + ':' + str(a[3]) + ':' + a[5] + ':' + str(a[0] - abns_k[max(n)][1]) + ':' + out_nge

			anchor_str = 'l'*anchored_left+'r'*anchored_right

			seq_name  = ''
			seq_name += k + DELIM_MAJOR												# read name
			seq_name += DELIM_MINOR.join([abns_k[m][2]+':'+abns_k[m][5] for m in n]) + DELIM_MAJOR	# viral refs this read spans
			#seq_name += DELIM_MINOR.join([str(m) for m in n]) + DELIM_MAJOR		# aln num of viral alignments (out of all alignments in the read)
			seq_name += anchor_str										# anchored on left, right, or both?
			if len(anchorseq_left):
				seq_name += DELIM_MAJOR + anchorseq_left
			if len(anchorseq_right):
				seq_name += DELIM_MAJOR + anchorseq_right

			seq_dat  = READDAT_BY_RNAME[k][abns_k[n[0]][0]:abns_k[n[-1]][1]]
			if len(seq_dat) >= 400:
				gk = ','.join(affected_genes)
				if gk not in vseq_out:
					vseq_out[gk] = []
				vseq_out[gk].append('>'+seq_name+'\n')
				vseq_out[gk].append(seq_dat+'\n')

		#
		#	FILTERING?
		#

		# [readPos1, readPos2, ref, pos1, pos2, orientation, mapq]
		subtel_dat = []
		type_dat   = []
		for i in range(len(abns_k)):
			[readPos1, readPos2, ref, pos1, pos2, orientation, mapq] = abns_k[i]
			subtel_dat.append(is_subtel(ref, [pos1, pos2]))
			if 'tel' in ref:
				if ref[-1] == '?':
					type_dat.append('tel?')
				else:
					type_dat.append('tel')
				if ref not in tel_count:
					tel_count[ref] = 0
				tel_count[ref] += 1
			else:
				if subtel_dat[-1] != None:
					type_dat.append('sub')
				else:
					type_dat.append('seq')

		#
		my_case = None

		# skip: ALL TEL
		if my_case == None and all([n[:3] == 'tel' for n in type_dat]):
			my_tel_len = sum([abs(n[1]-n[0]) for n in abns_k])
			if 'unanchored' not in ANCHORED_TEL:
				ANCHORED_TEL['unanchored'] = []
			ANCHORED_TEL['unanchored'].append(my_tel_len)
			ALL_TEL_LENS.append(my_tel_len)
			skip_count['only_tel'] += 1
			my_case = CASE_NUMBER_TELONLY

			if READS_ARE_PBSIM:
				conf_key = (k.split('-')[1].replace('_',''), 'unanchored')
				if conf_key not in CONF_DAT:
					CONF_DAT[conf_key] = 0
				CONF_DAT[conf_key] += 1

		# skip: TEL ON BOTH FLANKS
		if my_case == None and type_dat[0][:3] == 'tel' and type_dat[-1][:3] == 'tel':
			skip_count['2_tel_flank'] += 1
			my_case = CASE_NUMBER_DISCARD

		####print_me = False
		####for i in range(len(abns_k)):
		####	#if abns_k[i][2] == 'tel6p':
		####	#	print_me = True
		####	if abns_k[i][2] == 'chr8' and subtel_dat[i] != None and subtel_dat[i][1] == 'p':
		####		print_me = True
		####if print_me:
		####	for i in range(len(abns_k)):
		####		print(i, abns_k[i], subtel_dat[i])
		####	print('')
		####continue

		# annotate according to case type
		pick_anchor_from_inds = []
		get_tel_len_from_inds = []
		anchor_orientation    = None
		my_anchor_pos         = None
		tel_bases_in_seq      = None
		tel_frac              = 0.

		if my_case != CASE_NUMBER_DISCARD and my_case != CASE_NUMBER_TELONLY:
			# collapse all adjacent TEL / SUBTEL / SEQ into single spans
			current_start = 0
			current_type  = type_dat[0]
			type_ranges   = []
			for i in range(1,len(type_dat)):
				if is_same_type(type_dat[i], current_type) == False:	# merge tel? into adjacent tel
					type_ranges.append([current_start, i-1, current_type])
					current_start = i
					current_type  = type_dat[i]
			type_ranges.append([current_start, len(type_dat)-1, current_type])
			collapsed_types = [n[2] for n in type_ranges]

			#print(type_ranges)
			#for i in range(len(abns_k)):
			#	print(i, abns_k[i], subtel_dat[i])
			#print('')
			#continue

			# case 1a & 1b
			if collapsed_types == ['sub', 'tel']:
				pick_anchor_from_inds = [0]
				get_tel_len_from_inds = [1]
				anchor_orientation    = 'right'
				my_case = 1
			elif collapsed_types == ['tel', 'sub']:
				pick_anchor_from_inds = [1]
				get_tel_len_from_inds = [0]
				anchor_orientation    = 'left'
				my_case = 1

			# case 2a & 2b
			elif collapsed_types == ['seq', 'tel']:
				pick_anchor_from_inds = [0]
				get_tel_len_from_inds = [1]
				anchor_orientation    = 'right'
				my_case = 2
			elif collapsed_types == ['tel', 'seq']:
				pick_anchor_from_inds = [1]
				get_tel_len_from_inds = [0]
				anchor_orientation    = 'left'
				my_case = 2

			# case 3a & 3b
			elif collapsed_types == ['seq', 'sub', 'tel']:
				pick_anchor_from_inds = [0,1]
				get_tel_len_from_inds = [2]
				anchor_orientation    = 'right'
				my_case = 3
			elif collapsed_types == ['tel', 'sub', 'seq']:
				pick_anchor_from_inds = [1,2]
				get_tel_len_from_inds = [0]
				anchor_orientation    = 'left'
				my_case = 3

			# case 4a & 4b
			elif collapsed_types == ['tel', 'seq', 'sub'] or collapsed_types == ['sub', 'seq', 'tel']:
				my_seq_range     = list(range(type_ranges[1][0],type_ranges[1][1]+1))
				seq_len_sum      = sum([abs(abns_k[n][3] - abns_k[n][4]) for n in my_seq_range])
				#(rs1, rs2)      = (abns_k[my_seq_range[0]][0], abns_k[my_seq_range[-1]][1])
				rs1 = min([abns_k[n][0] for n in my_seq_range])
				rs2 = max([abns_k[n][1] for n in my_seq_range])
				tel_bases_in_seq = count_tel_repeat_bases_in_string(READDAT_BY_RNAME[k][rs1:rs2])
				tel_frac         = tel_bases_in_seq/float(rs2-rs1)
				#
				my_case = CASE_NUMBER_DISCARD
				if collapsed_types == ['tel', 'seq', 'sub']:
					my_sub_range = list(range(type_ranges[2][0],type_ranges[2][1]+1))
					sub_len_max  = max([abs(abns_k[n][3] - abns_k[n][4]) for n in my_sub_range])
					if sub_len_max/float(seq_len_sum) >= MIN_SUB_TO_SEQ_LEN_RATIO:
						pick_anchor_from_inds = [2]
						get_tel_len_from_inds = [0]
						anchor_orientation    = 'left'
						my_case = 4
				elif collapsed_types == ['sub', 'seq', 'tel']:
					my_sub_range = list(range(type_ranges[0][0],type_ranges[0][1]+1))
					sub_len_max  = max([abs(abns_k[n][3] - abns_k[n][4]) for n in my_sub_range])
					if sub_len_max/float(seq_len_sum) >= MIN_SUB_TO_SEQ_LEN_RATIO:
						pick_anchor_from_inds = [0]
						get_tel_len_from_inds = [2]
						anchor_orientation    = 'right'
						my_case = 4

			# case 5a & 5b
			elif collapsed_types == ['seq', 'tel', 'sub'] or collapsed_types == ['sub', 'tel', 'seq']:
				if collapsed_types == ['seq', 'tel', 'sub']:
					my_seq_range = list(range(type_ranges[0][0],type_ranges[0][1]+1))
					my_sub_range = list(range(type_ranges[2][0],type_ranges[2][1]+1))
				elif collapsed_types == ['sub', 'tel', 'seq']:
					my_seq_range = list(range(type_ranges[2][0],type_ranges[2][1]+1))
					my_sub_range = list(range(type_ranges[0][0],type_ranges[0][1]+1))
				seq_len_sum = sum([abs(abns_k[n][3] - abns_k[n][4]) for n in my_seq_range])
				sub_len_max = max([abs(abns_k[n][3] - abns_k[n][4]) for n in my_sub_range])
				#
				my_case = CASE_NUMBER_DISCARD
				if sub_len_max/float(seq_len_sum) >= MIN_SUB_TO_SEQ_LEN_RATIO:
					#(rs1, rs2)      = (abns_k[my_seq_range[0]][0], abns_k[my_seq_range[-1]][1])
					rs1 = min([abns_k[n][0] for n in my_seq_range])
					rs2 = max([abns_k[n][1] for n in my_seq_range])
					tel_bases_in_seq = count_tel_repeat_bases_in_string(READDAT_BY_RNAME[k][rs1:rs2])
					tel_frac         = tel_bases_in_seq/float(rs2-rs1)
					if collapsed_types == ['seq', 'tel', 'sub']:
						pick_anchor_from_inds = [2]
						get_tel_len_from_inds = [1]
						anchor_orientation    = 'left'
					elif collapsed_types == ['sub', 'tel', 'seq']:
						pick_anchor_from_inds = [0]
						get_tel_len_from_inds = [1]
						anchor_orientation    = 'right'
					my_case = 5

			# skip: TEL ON NO FLANKS ("tel?" doesn't count here)
			elif type_dat[0] != 'tel' and type_dat[-1] != 'tel':
				skip_count['0_tel_flank'] += 1
				my_case = CASE_NUMBER_DISCARD

			# too hard, lets give up..
			else:
				my_case = CASE_NUMBER_COMPLEX

		#
		# tabulate length & determine anchor
		#
		my_anchor  = None
		my_tel_len = 0
		if len(get_tel_len_from_inds):
			my_tel_range = []
			for i in get_tel_len_from_inds:
				my_tel_range.extend(list(range(type_ranges[i][0],type_ranges[i][1]+1)))
			my_tel_pos_dict = {}
			for i in my_tel_range:
				s = sorted([abns_k[i][0], abns_k[i][1]])
				for j in range(s[0],s[1]):
					my_tel_pos_dict[j] = True
			my_tel_len = len(my_tel_pos_dict)
			if tel_bases_in_seq != None and tel_frac >= MIN_TEL_FRAC_TO_ADD_TELSEQ_BASES:
				my_tel_len += tel_bases_in_seq

			my_anc_range = []
			for i in pick_anchor_from_inds:
				my_anc_range.extend(list(range(type_ranges[i][0],type_ranges[i][1]+1)))
			my_anchors = []
			for i in my_anc_range:
				my_anchors.append([abs(abns_k[i][1] - abns_k[i][0]), abns_k[i][2], subtel_dat[i], i])
			my_anchors = sorted(my_anchors, reverse=True)
			if anchor_orientation == 'left':
				apos = abns_k[my_anchors[0][3]][3]
			elif anchor_orientation == 'right':
				apos = abns_k[my_anchors[0][3]][4]
			else:
				print('how did you end up here??', anchor_orientation)
				exit(1)

			my_anchor     = my_anchors[0][1]
			my_anchor_pos = (my_anchor, apos)
			if my_anchors[0][2] != None:
				my_anchor += my_anchors[0][2][1]
			else:
				ai = abns_k[my_anchors[0][3]]
				relax_subtel = is_subtel(ai[2], [ai[3], ai[4]], relaxed=True)
				if relax_subtel != None:
					my_anchor += relax_subtel[1]
				else:
					# well we tried...
					if my_anchor not in CHR_FAILED_TO_ANCHOR:
						CHR_FAILED_TO_ANCHOR[my_anchor] = 0
					CHR_FAILED_TO_ANCHOR[my_anchor] += 1
					#my_anchor = 'unanchored'
					my_case = CASE_NUMBER_DISCARD
					continue

			if my_anchor not in ANCHORED_TEL:
				ANCHORED_TEL[my_anchor] = []
			ANCHORED_TEL[my_anchor].append(my_tel_len)

			if my_anchor != 'unanchored':
				if my_anchor not in ANCHORED_POS:
					ANCHORED_POS[my_anchor] = []
				ANCHORED_POS[my_anchor].append(my_anchor_pos[1])

			if READS_ARE_PBSIM:
				conf_key = (k.split('-')[1].replace('_',''), my_anchor)
				if conf_key not in CONF_DAT:
					CONF_DAT[conf_key] = 0
				CONF_DAT[conf_key] += 1

			#print(my_case, type_ranges)
			#print(my_anchors)
			#print(my_anchor, my_anchor_pos)
			#print('')

		#
		if my_case not in case_count:
			case_count[my_case] = 0
		case_count[my_case] += 1

		#
		#	PLOTTING
		#

		BN = -0.05 * READLEN_BY_RNAME[k]
		BP =  0.05 * READLEN_BY_RNAME[k]

		y_offsets = [0.7*BP]

		polygons = []
		p_alpha  = []
		p_color  = []
		p_text   = []
		plot_me  = False
		for i in range(len(ALIGNMENTS_BY_RNAME[k])):
			n = abns_k[i]
			p_alpha.append(float(n[6]+15.)/(60.+15.))

			my_refname = n[2]
			my_refspan = str(n[3])+' - '+str(n[4])

			if 'tel' in my_refname:
				if my_refname[-1] == '?':	# skip these weird 
					p_color.append('gray')
				else:
					p_color.append('red')
					plot_me = True

			elif subtel_dat[i] != None:
				p_color.append('purple')
				my_refname = 'sub' + my_refname[3:] + subtel_dat[i][1]
				my_refspan = '[' + str(subtel_dat[i][0]) + ']'

			else:
				p_color.append('blue')

			delta_pointy     = 0.8*(n[1]-n[0])
			delta_pointy_rev = 0.2*(n[1]-n[0])

			if n[5] == 'FWD':
				polygons.append(Polygon(np.array([[n[0],BN], [n[0],BP], [n[0]+delta_pointy,BP], [n[1],0.], [n[0]+delta_pointy,BN]]), closed=True))
			else:
				polygons.append(Polygon(np.array([[n[0]+delta_pointy_rev,BN], [n[0],0.], [n[0]+delta_pointy_rev,BP], [n[1],BP], [n[1],BN]]), closed=True))
			
			p_text.append((n[0], y_offsets[-1], my_refname+' : '+my_refspan))
			y_offsets.append(y_offsets[-1] - 0.40*BP)

		#
		if plot_me == False or SKIP_READ_PLOTS:
			continue

		fig = mpl.figure(0,figsize=(12,6))
		ax = mpl.gca()
		for i in range(len(polygons)):
			ax.add_collection(PatchCollection([polygons[i]], alpha=p_alpha[i], color=p_color[i]))
		for i in range(len(p_text)):
			mpl.text(p_text[i][0], p_text[i][1], p_text[i][2], ha='left', fontsize=9)

		mpl.axis([0, READLEN_BY_RNAME[k]+1, 4.0*BN, 4.0*BP])
		mpl.yticks([],[])
		mpl.grid(linestyle='--', alpha=0.5)
		mpl.xlabel('read coordinates')
		mpl.title(k + ' --- tel_len: ' + str(my_tel_len))
		mpl.tight_layout()

		if my_case == CASE_NUMBER_TELONLY:
			my_plot_dir = PLOT_DIR + 'other-unanchored/'
		elif my_case == CASE_NUMBER_COMPLEX:
			my_plot_dir = PLOT_DIR + 'other-complex/'
		elif my_case == CASE_NUMBER_DISCARD:
			my_plot_dir = PLOT_DIR + 'other-discard/'
		else:
			my_plot_dir = PLOT_DIR + my_anchor + '/'
		makedir(my_plot_dir)
		mpl.savefig(my_plot_dir + 'read_' + str(nPlot) + '.png')

		nPlot += 1
		mpl.close(fig)
	f_out.close()

	#
	#	WRITE TEL SEQUENCES
	#
	f_out = open(TEL_FASTA, 'w')
	for k in sorted(vseq_out.keys()):
		for n in vseq_out[k]:
			f_out.write(n)
	f_out.close()

	#
	#	WRITE TEL LEN DICT
	#
	f_out = open(TEL_JSON, 'w')
	json.dump(ANCHORED_TEL, f_out, sort_keys=True, indent=4)
	f_out.close()

else:
	ANCHORED_TEL = ANCHORED_TEL_JSON	# skip everything, use precomputed tel lens

#
readcount_denom = max([len(ANCHORED_TEL[k]) for k in ANCHORED_TEL.keys() if k != 'unanchored'])
width_max       = 1.0
width_min       = 0.1

#
#	PLOT WHOLE-GENOME SUMMARY (SCATTER)
#
xlab = ['-'] + [str(n) for n in range(1,22+1)] + ['X']
xtck = list(range(1,len(xlab)+1))

ref_2_x = {'chr'+xlab[i]:xtck[i] for i in range(len(xlab))}
ref_2_x['unanchored'] = xtck[0]
ref_2_x['unanchore']  = xtck[0]

dat_x = []
dat_y = []
dat_c = []
for k in ANCHORED_TEL.keys():
	for n in ANCHORED_TEL[k]:
		dat_x.append(ref_2_x[k[:-1]])
		if k[-1] == 'p' or k == 'unanchored':
			dat_y.append(n)
			dat_c.append('blue')
		elif k[-1] == 'q':
			dat_y.append(-n)
			dat_c.append('red')

Y_TICK = 5000
#y_min = ( (max([abs(n) for n in dat_y if n < 0]) // Y_TICK) + 1 ) * Y_TICK
#y_max = ( (max([abs(n) for n in dat_y if n > 0]) // Y_TICK) + 1 ) * Y_TICK
(y_min, y_max) = (20000, 20000)
ytck  = list(range(-y_min, y_max+Y_TICK, Y_TICK))
ylab  = []
for n in ytck:
	if n == 0:
		ylab.append('')
	else:
		ylab.append(str(abs(n)//1000) + 'k')

fig = mpl.figure(0,figsize=(16,6))
mpl.scatter(dat_x, dat_y, s=100, c=dat_c, linewidths=0)
mpl.plot([0,len(xlab)+1], [0,0], '-k', linewidth=3)
mpl.xticks(xtck, xlab)
mpl.xlim([0,len(xlab)+1])
mpl.yticks(ytck, ylab)
mpl.ylim([-y_min, y_max])
mpl.ylabel('<-- q   telomere length   p -->')
mpl.grid(linestyle='--', alpha=0.5)
mpl.tight_layout()
#mpl.show()
mpl.savefig(SUMMARY_SCATTER)

#
#	PLOT WHOLE-GENOME SUMMARY (VIOLIN)
#
(dat_l_p, dat_l_q) = ([], [])
(dat_p_p, dat_p_q) = ([], [])
(dat_w_p, dat_w_q) = ([], [])
for k in ANCHORED_TEL.keys():
	my_width = min( [width_max, max([width_min, width_max*(float(len(ANCHORED_TEL[k]))/readcount_denom)])] )
	if k[-1] == 'p' or k == 'unanchored':
		dat_p_p.append(ref_2_x[k[:-1]])
		dat_l_p.append([])
		dat_w_p.append(my_width)
	elif k[-1] == 'q':
		dat_p_q.append(ref_2_x[k[:-1]])
		dat_l_q.append([])
		dat_w_q.append(my_width)
	for n in ANCHORED_TEL[k]:
		if k[-1] == 'p' or k == 'unanchored':
			dat_l_p[-1].append(n)
		elif k[-1] == 'q':
			dat_l_q[-1].append(-n)

v_line_keys = ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians', 'cquantiles']
fig = mpl.figure(1,figsize=(16,6))

violin_parts_p = mpl.violinplot(dat_l_p, dat_p_p, points=200, widths=dat_w_p)
for pc in violin_parts_p['bodies']:
	pc.set_facecolor('blue')
	pc.set_edgecolor('black')
	pc.set_alpha(0.7)
for k in v_line_keys:
	if k in violin_parts_p:
		violin_parts_p[k].set_color('black')
		violin_parts_p[k].set_alpha(0.3)

violin_parts_q = mpl.violinplot(dat_l_q, dat_p_q, points=200, widths=dat_w_q)
for pc in violin_parts_q['bodies']:
	pc.set_facecolor('red')
	pc.set_edgecolor('black')
	pc.set_alpha(0.7)
for k in v_line_keys:
	if k in violin_parts_q:
		violin_parts_q[k].set_color('black')
		violin_parts_q[k].set_alpha(0.3)

if len(COMP_DICT) == 0:
	for i in range(len(dat_l_p)):
		yval = np.mean(dat_l_p[i])
		xval = dat_p_p[i]
		mpl.plot([xval - 0.3, xval + 0.3], [yval, yval], '-k', linewidth=2, alpha=0.4)
	for i in range(len(dat_l_q)):
		yval = np.mean(dat_l_q[i])
		xval = dat_p_q[i]
		mpl.plot([xval - 0.3, xval + 0.3], [yval, yval], '-k', linewidth=2, alpha=0.4)
else:
	for k in COMP_DICT.keys():
		xval = ref_2_x[k[:-1]]
		if k[-1] == 'p':
			yval = COMP_DICT[k]
		elif k[-1] == 'q':
			yval = -COMP_DICT[k]
		else:
			print('skipping weird benchmark contig:', k, COMP_DICT[k])
			continue
		mpl.plot([xval - 0.35, xval + 0.35], [yval, yval], '-k', linewidth=3, alpha=1.0)

mpl.plot([0,len(xlab)+1], [0,0], '-k', linewidth=3)
mpl.xticks(xtck, xlab)
mpl.xlim([0,len(xlab)+1])
mpl.yticks(ytck, ylab)
mpl.ylim([-y_min, y_max])
mpl.ylabel('<-- q   telomere length   p -->')
mpl.grid(linestyle='--', alpha=0.5)
mpl.tight_layout()
mpl.savefig(SUMMARY_VIOLIN)

#
#	ANCHOR POSITION PLOTS
#
print('plotting QC plots...')
BP_PER_BIN = 300
MIN_BINS   = 30
for k in ANCHORED_POS.keys():
	(pmin, pmax) = (min(ANCHORED_POS[k]), max(ANCHORED_POS[k]))
	nbins = max([MIN_BINS, int((pmax - pmin)/float(BP_PER_BIN) + 0.5)])

	fig = mpl.figure(2,figsize=(12,6))
	mpl.hist(ANCHORED_POS[k], bins=nbins)
	mpl.grid(linestyle='--', alpha=0.5)
	mpl.ylabel('read count')
	mpl.xlabel('position (' + k + ')')
	mpl.tight_layout()
	mpl.savefig(QCPLOT_DIR + k + '.png')
	mpl.close(fig)

#
#	CONFUSION MATRIX (FOR PBSIM SIMULATED READS)
#
if READS_ARE_PBSIM:
	ref_2_ci = {}
	clab     = []
	for ci in xlab:
		if ci == '-':
			ref_2_ci['-'] = 0
			clab.append('-')
		else:
			ref_2_ci['chr' + ci + 'p'] = len(ref_2_ci)
			ref_2_ci['chr' + ci + 'q'] = len(ref_2_ci)
			clab.append('chr' + ci + 'p')
			clab.append('chr' + ci + 'q')
	ref_2_ci['unanchored'] = 0

	# (truth, where_we_ended_up)
	Z = [[1,2,3],[4,5,6],[7,8,9]]
	Z = [[0 for n in range(len(ref_2_ci))] for m in range(len(ref_2_ci))]
	for k1 in sorted(CONF_DAT.keys()):
		#print(k1, CONF_DAT[k1], ref_2_ci[k1[0]], ref_2_ci[k1[1]])
		Z[ref_2_ci[k1[0]]][ref_2_ci[k1[1]]] = CONF_DAT[k1]

	mpl.rcParams.update({'font.size': 14, 'font.weight':'bold'})

	fig = mpl.figure(3,figsize=(12,10))
	Z = np.array(Z[::-1])
	X, Y = np.meshgrid( range(0,len(Z[0])+1), range(0,len(Z)+1) )
	mpl.pcolormesh(X,Y,Z)
	mpl.axis([0,len(Z[0]),0,len(Z)])
	mpl.yticks(np.arange(0,len(clab))+1.5, clab[::-1])
	mpl.xticks(np.arange(0,len(clab))+0.5, clab, rotation=90)
	mpl.grid(linestyle='--', alpha=0.5)
	mpl.title('subtel confusion matrix')
	mpl.ylabel('ground truth contig')
	mpl.xlabel('where we ended up')
	cb = mpl.colorbar()
	cb.set_label('# reads')
	mpl.tight_layout()
	mpl.savefig(CONFUSION_PLOT)

#
#	PRINT OUTPUT STATS AND ETC
#
if USE_JSON == False:
	#for k in sorted(tel_count.keys()):
	#	print(k, tel_count[k])
	
	print('')
	for k in sorted(ANCHORED_TEL.keys()):
		#print(k, ANCHORED_TEL[k])
		print(k, int(np.mean(ANCHORED_TEL[k])), int(np.median(ANCHORED_TEL[k])))
	print('')
	
	print('CASE TYPES:')
	print('===========')
	for k in sorted(case_count.keys()):
		print(k, case_count[k])
	print('')
	
	print('READS SKIPPED:')
	print('==============')
	for k in sorted(skip_count.keys()):
		print(k, skip_count[k])
	print('')
	
	print('FAILED ANCHORS:')
	print('===============')
	for k in sorted(CHR_FAILED_TO_ANCHOR.keys()):
		print(k, CHR_FAILED_TO_ANCHOR[k])
	print('')


