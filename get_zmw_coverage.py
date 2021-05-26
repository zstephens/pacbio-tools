#!/usr/bin/env python
import os
import re
import sys
import argparse
import numpy as np

from common.mappability_corgi import MappabilityTrack
from common.ref_func_corgi import HG38_SIZES

# samtools view input.bam | python3 get_zmw_coverage.py <args>

REF_CHAR = 'MX=D'

def strip_polymerase_coords(rn):
	return '/'.join(rn.split('/')[:-1])

def is_valid_coord(my_chr, my_pos, bed_list=[]):
	if any([n.query(my_chr,my_pos) for n in bed_list]):
		return False
	return True

def reads_2_cov(my_chr, readpos_list_all, out_dir, bed_list=[]):
	if my_chr not in HG38_SIZES:
		print('skipping '+my_chr+'...')
		return None
	denom  = HG38_SIZES[my_chr]
	for i in xrange(len(bed_list)):
		track = bed_list[i].all_tracks[my_chr]
		for j in xrange(1, len(track)-1, 2):
			denom -= track[j+1]-track[j]
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
		for i in xrange(len(readpos_list)):
			biggest_rlen = max([biggest_rlen, readpos_list[i][1]-readpos_list[i][0]])
		readlens.append(biggest_rlen)
		#print(readpos_list, len(readpos_list), '-->',)
		found_overlaps = True
		while found_overlaps:
			found_overlaps = False
			for i in xrange(len(readpos_list)):
				for j in xrange(i+1,len(readpos_list)):
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
		#
		for rspan in readpos_list:
			cov[rspan[0]:rspan[1]] += 1
	# write output
	if out_dir[-1] != '/':
		out_dir += '/'
	out_file = out_dir + 'zmw-coverage_' + my_chr + '.dat'
	cov.tofile(out_file)
	print('mean zmw cov:        ', '{0:0.3f}'.format(np.sum(cov)/float(denom)))
	print('mean aligned readlen:', int(np.mean(readlens)))

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
tlen_by_zmw = []	# max tlen observed for each zmw

for line in input_stream:
	if len(line) and line[0] != '#':
		splt  = line.strip().split('\t')
		ref   = splt[2]
		pos   = int(splt[3])
		cigar = splt[5]

		if READ_MODE == 'CLR':
			rnm = strip_polymerase_coords(splt[0])
			template_len = splt[0].split('/')[-1].split('_')
			template_len = int(template_len[1]) - int(template_len[0])
		elif READ_MODE == 'CCS':
			rnm = splt[0]
			template_len = len(splt[9])

		if ref != prev_ref:
			print('processing reads on '+ref+'...')
			# output stuff
			if prev_ref != None:
				reads_2_cov(ref, alns_by_zmw, OUT_PATH, EXCL_BED)
			alns_by_zmw = []
			rnm_dict = {}
			prev_ref = ref

		letters = re.split(r"\d+",cigar)[1:]
		numbers = [int(n) for n in re.findall(r"\d+",cigar)]
		adj     = 0
		for i in xrange(len(letters)):
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
			tlen_by_zmw.append(0)

		alns_by_zmw[my_rind].append((pos, pos+adj))
		tlen_by_zmw[my_rind] = max([tlen_by_zmw[my_rind], template_len])

print('mean template length:  ', int(np.mean(tlen_by_zmw)))
print('median template length:', int(np.median(tlen_by_zmw)))
