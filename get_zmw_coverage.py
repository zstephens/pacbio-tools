#!/usr/bin/env python
import os
import re
import sys
import argparse
import numpy as np

from common.mappability_corgi import MappabilityTrack
from common.ref_func_corgi import HG38_SIZES, LEXICO_2_IND, makedir

# samtools view input.bam | python3 get_zmw_coverage.py <args>

REF_CHAR = 'MX=D'

def strip_polymerase_coords(rn):
	return '/'.join(rn.split('/')[:-1])

def is_valid_coord(my_chr, my_pos, bed_list=[]):
	if any([n.query(my_chr,my_pos) for n in bed_list]):
		return False
	return True

#
# returns (total_bases_mapped, avg_cov, total_uncov_pos, fraction_uncov_pos, nonexcluded_pos)
#
def reads_2_cov(my_chr, readpos_list_all, out_dir, bed_list=[]):
	if my_chr not in HG38_SIZES:
		print('skipping '+my_chr+'...')
		return None
	denom   = HG38_SIZES[my_chr]
	oob_pos = 0						# num out-of-bounds positions
	for i in range(len(bed_list)):
		if my_chr in bed_list[i].all_tracks:
			track = bed_list[i].all_tracks[my_chr]
			for j in range(1, len(track)-1, 2):
				denom   -= track[j+1]-track[j]
				oob_pos += track[j+1]-track[j]
			#print(HG38_SIZES[my_chr], '-->', denom)
	#cov = np.zeros(HG38_SIZES[my_chr] , dtype='<B')
	cov = np.zeros(HG38_SIZES[my_chr] , dtype='<i4')

	readlens = []
	print('computing zmw coverage on '+my_chr+'...')
	# collapse overlapping alignments
	for readpos_list in readpos_list_all:
		if len(readpos_list_all) == 0:
			continue
		biggest_rlen = 0
		for i in range(len(readpos_list)):
			biggest_rlen = max([biggest_rlen, readpos_list[i][1]-readpos_list[i][0]])
		readlens.append(biggest_rlen)
		#print(readpos_list, len(readpos_list), '-->',)
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
		#print(len(readpos_list), readpos_list)
		for rspan in readpos_list:
			cov[rspan[0]:rspan[1]] += 1
	
	# write output
	if out_dir[-1] != '/':
		out_dir += '/'
	out_fn   = 'zmw-coverage_' + my_chr
	out_file = out_dir + out_fn + '.npz'
	#cov.tofile(out_file)	# takes up too much space
	print('saving ' + out_fn + '.npz...')
	np.savez_compressed(out_file, cov, out_fn)

	# stats
	total_bases = np.sum(cov)
	avg_cov     = total_bases/float(denom)
	total_unmap = len(cov) - np.count_nonzero(cov) - oob_pos
	avg_unmap   = total_unmap/float(denom)
	print('mean zmw cov:        ', '{0:0.3f}'.format(avg_cov))
	print('frac uncovered pos:  ', '{0:0.3f}'.format(avg_unmap))
	print('mean aligned readlen:', int(np.mean(readlens)))
	return (total_bases, avg_cov, total_unmap, avg_unmap, denom)

#
# parse input args
#
parser = argparse.ArgumentParser(description='get_zmw_coverage.py')
parser.add_argument('-m', type=str, required=True,  metavar='<str>', help="* mode (CCS/CLR)")
parser.add_argument('-o', type=str, required=True,  metavar='<str>', help="* /path/to/output/dir/")
parser.add_argument('-b', type=str, required=False, metavar='<str>', help="/path/to/bed/dir/", default=None)
args = parser.parse_args()

READ_MODE = args.m
if READ_MODE not in ['CCS', 'CLR']:
	print('Error: Unknown read mode.')
	exit(1)

OUT_PATH = args.o
if OUT_PATH[-1] != '/':
	OUT_PATH += '/'
makedir(OUT_PATH)
OUT_REPORT = OUT_PATH+'cov_report.tsv'

RESOURCE_PATH = args.b
if RESOURCE_PATH == None:
	SIM_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'
	RESOURCE_PATH = SIM_PATH + 'resources/'
if RESOURCE_PATH[-1] != '/':
	RESOURCE_PATH += '/'
EXCL_BED = [MappabilityTrack(RESOURCE_PATH + 'hg38_centromere-and-gap_unsorted.bed', bed_buffer=1000)]

#
#
#
if not sys.stdin.isatty():
	input_stream = sys.stdin
else:
	print('No input.')
	exit(1)

prev_ref = None
rnm_dict = {}
alns_by_zmw = []	# alignment start/end per zmw
rlen_by_zmw = []	# max tlen observed for each zmw
covdat_by_ref = {}	#

for line in input_stream:
	if len(line) and line[0] != '#':
		splt  = line.strip().split('\t')
		ref   = splt[2]
		pos   = int(splt[3])
		cigar = splt[5]

		if ref == '*':		# skip unmapped reads
			break			# it's safe to quit at this point, right?

		#if pos > 1000000:	# for debugging purposes
		#	continue

		if READ_MODE == 'CLR':
			rnm = strip_polymerase_coords(splt[0])
			template_len = splt[0].split('/')[-1].split('_')
			template_len = int(template_len[1]) - int(template_len[0])
		elif READ_MODE == 'CCS':
			rnm = splt[0]
			template_len = len(splt[9])

		if ref != prev_ref:
			# compute coverage on previous ref now that we're done
			if prev_ref != None and len(alns_by_zmw) and prev_ref in HG38_SIZES:
				covdat_by_ref[prev_ref] = reads_2_cov(prev_ref, alns_by_zmw, OUT_PATH, EXCL_BED)
			# reset for next ref
			if ref in HG38_SIZES:
				print('processing reads on '+ref+'...')
			else:
				print('skipping reads on '+ref+'...')
			alns_by_zmw = []
			rnm_dict = {}
			prev_ref = ref

		if ref not in HG38_SIZES:
			continue

		letters = re.split(r"\d+",cigar)[1:]
		numbers = [int(n) for n in re.findall(r"\d+",cigar)]
		adj     = 0
		for i in range(len(letters)):
			if letters[i] in REF_CHAR:
				adj += numbers[i]

		# skip telomeres + centromeres + gaps
		if is_valid_coord(ref,pos,EXCL_BED) == False or is_valid_coord(ref,pos+adj,EXCL_BED) == False:
			continue

		if rnm in rnm_dict:
			my_rind = rnm_dict[rnm]
		else:
			rnm_dict[rnm] = len(rnm_dict)
			my_rind       = len(rnm_dict)-1
			alns_by_zmw.append([])
			rlen_by_zmw.append(0)

		alns_by_zmw[my_rind].append((pos, pos+adj))
		rlen_by_zmw[my_rind] = max([rlen_by_zmw[my_rind], template_len])

# we probably we need to process the final ref, assuming no contigs beyond chrM
if ref not in covdat_by_ref and len(alns_by_zmw) and ref in HG38_SIZES:
	covdat_by_ref[ref] = reads_2_cov(ref, alns_by_zmw, OUT_PATH, EXCL_BED)

# sum up data across chromosomes to get whole-genome stats
all_mapped_bases = 0
all_uncovered    = 0
all_denom        = 0
for k in covdat_by_ref.keys():
	if k != None:
		all_mapped_bases += covdat_by_ref[k][0]
		all_uncovered    += covdat_by_ref[k][2]
		all_denom        += covdat_by_ref[k][4]
all_avg_cov    = all_mapped_bases/float(all_denom)
all_unmap_frac = all_uncovered/float(all_denom)

# remove 0-len tlens that shouldn't be here but are for some reason
rlen_by_zmw = [n for n in rlen_by_zmw if n > 0]

# final output report
sorted_refs = [n[1] for n in sorted([(LEXICO_2_IND[k],k) for k in covdat_by_ref.keys()])]
f = open(OUT_REPORT, 'w')
f.write('\t'.join(['CHR', 'MAPPED_BASES', 'AVG_COV', 'UNMAPPED_BASES', 'UNMAPPED_FRAC']) + '\n')
f.write('\t'.join(['ALL', str(all_mapped_bases), '{0:0.3f}'.format(all_avg_cov), str(all_uncovered), '{0:0.3f}'.format(all_unmap_frac)]) + '\n')
for k in sorted_refs:
	cd = covdat_by_ref[k]
	f.write('\t'.join([k, str(cd[0]), '{0:0.3f}'.format(cd[1]), str(cd[2]), '{0:0.3f}'.format(cd[3])]) + '\n')
f.close()

# stats
print('')
print('mean read length:      ', int(np.mean(rlen_by_zmw)))
print('median read length:    ', int(np.median(rlen_by_zmw)))
if 'chrX' in covdat_by_ref and 'chrY' in covdat_by_ref:
	print('chrX:chrY mapped bases:', '{0:0.3f}'.format(covdat_by_ref['chrX'][0]/float(covdat_by_ref['chrY'][0])))
if 'chrX' in covdat_by_ref and 'chr2' in covdat_by_ref:
	print('chrX:chr2 mapped bases:', '{0:0.3f}'.format(covdat_by_ref['chrX'][0]/float(covdat_by_ref['chr2'][0])))
if 'chrM' in covdat_by_ref and 'chr2' in covdat_by_ref:
	print('chrM:chr2 mapped bases:', '{0:0.3f}'.format(covdat_by_ref['chrM'][0]/float(covdat_by_ref['chr2'][0])))
print('')
