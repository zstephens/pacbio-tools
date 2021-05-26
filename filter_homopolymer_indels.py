import os
import re
import argparse

from common.ref_func_corgi import index_ref, read_ref_py3, exists_and_is_nonZero, LEXICO_2_IND

def get_char_count(s):
	out_dict = {}
	for n in s:
		if n not in out_dict:
			out_dict[n] = 0
		out_dict[n] += 1
	return out_dict

def main(raw_args=None):
	parser = argparse.ArgumentParser(description='filter_homopolymer_indels.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
	parser.add_argument('-r', type=str, required=True,  default=argparse.SUPPRESS, metavar='<ref.fa>',     help="* Path to reference fasta")
	parser.add_argument('-v', type=str, required=True,  default=argparse.SUPPRESS, metavar='<input.vcf>',  help="* Path to input vcf")
	parser.add_argument('-o', type=str, required=True,  default=argparse.SUPPRESS, metavar='<output.vcf>', help="* Path to output vcf")
	parser.add_argument('-d', type=int, required=False, default=10,                metavar='<int>',        help="Minimum DP")
	parser.add_argument('-l', type=int, required=False, default=3,                 metavar='<int>',        help="Filter all indels in homopolymers > this length")
	parser.add_argument('--pass-only',  required=False, default=False,             action='store_true',    help='Only output PASS variants')
	args = parser.parse_args(raw_args)

	IN_REF  = args.r
	IN_VCF  = args.v
	OUT_VCF = args.o

	MIN_DP    = args.d
	MAX_HOMO  = args.l
	PASS_ONLY = args.pass_only

	# index ref
	if exists_and_is_nonZero(IN_REF):
		ref_inds = index_ref(IN_REF)
		ri_rev   = {ref_inds[n][0]:n for n in range(len(ref_inds))}
	else:
		print('Reference fasta not found.')
		exit(1)

	# open vcfs
	if exists_and_is_nonZero(IN_VCF):
		f_in = open(IN_VCF, 'r')
	else:
		print('Input vcf not found.')
		exit(1)
	f_out = open(OUT_VCF, 'w')

	# organize input variants and write header
	var_dat = []
	for line in f_in:
		if line[0] == '#':	# header
			f_out.write(line)
		else:
			splt = line.strip().split('\t')
			var_dat.append([LEXICO_2_IND[splt[0]], int(splt[1]), [n for n in splt]])
	var_dat = [n[2] for n in sorted(var_dat)]

	#
	# analyze variants...
	#
	TOTAL_INPUT_VARIANTS  = 0
	TOTAL_OUTPUT_VARIANTS = 0
	excl_pass   = 0
	excl_dp     = 0
	excl_homo   = 0
	prev_ref    = None
	current_seq = None
	for variant in var_dat:
		TOTAL_INPUT_VARIANTS += 1
		# basic qual filters
		if PASS_ONLY and variant[6] != 'PASS':
			excl_pass += 1
			continue
		my_info = variant[7] + ';'
		my_dp   = re.findall(r"DP=.*?(?=;)", my_info)[0][3:]
		if int(my_dp) < MIN_DP:
			excl_dp += 1
			continue
		# write SNVs / MNVs directly to output
		var_ref = variant[3]
		var_alt = variant[4]
		if len(var_ref) == len(var_alt):
			TOTAL_OUTPUT_VARIANTS += 1
			f_out.write('\t'.join(variant) + '\n')
			continue
		# read in ref seq
		my_ref = variant[0]
		if my_ref != prev_ref:
			my_ri       = ri_rev[my_ref]
			current_seq = read_ref_py3(IN_REF, ref_inds[my_ri])
			prev_ref    = my_ref
		# check if sequence alteration caused by variant involves only a single nucleotide
		my_pos = int(variant[1]) - 1	# 1-index --> 0-index
		if len(var_ref) > len(var_alt):
			delta_seq = var_ref[len(var_alt):]
		else:
			delta_seq = var_alt[len(var_ref):]
		char_count = get_char_count(delta_seq)
		if len(char_count) > 1:
			TOTAL_OUTPUT_VARIANTS += 1
			f_out.write('\t'.join(variant) + '\n')
			continue
		# check homopolymers to the left
		h_l = my_pos
		while current_seq[h_l-1:h_l] == var_alt[0]:	# slicing used to ensure string, not ord
			h_l -= 1
		homoseq_l = current_seq[h_l:my_pos]
		# check homopolymers to the right
		h_r = my_pos + len(var_ref)
		while current_seq[h_r:h_r+1] == var_alt[-1]:	# slicing used to ensure string, not ord
			h_r += 1
		homoseq_r = current_seq[my_pos+len(var_ref):h_r]
		# see if length of homopolymer is above threshold
		if len(homoseq_l) and len(homoseq_r) and homoseq_l[0] == homoseq_r[0]:
			my_homo_len = len(homoseq_l) + len(homoseq_r)
		else:
			my_homo_len = max([len(homoseq_l), len(homoseq_r)])
		if my_homo_len > MAX_HOMO:
			excl_homo += 1
			continue
		# looks like we're ok!
		TOTAL_OUTPUT_VARIANTS += 1
		f_out.write('\t'.join(variant) + '\n')

		## debug
		#print variant, my_dp, delta_seq
		#print current_seq[my_pos-10:my_pos], current_seq[my_pos:my_pos+len(var_ref)], current_seq[my_pos+len(var_ref):my_pos+len(var_ref)+10]
		#print 'left: ', homoseq_l
		#print 'right:', homoseq_r
		#print 'total:', my_homo_len
		#print ''

	# close files
	f_out.close()
	f_in.close()

	OUT_STRING_SIZE = 23
	OUT_PRINT = [['INPUT_VARS:', TOTAL_INPUT_VARIANTS],
	             ['# excluded (PASS):', excl_pass],
	             ['# excluded (DP < ' + str(MIN_DP) + '):', excl_dp],
	             ['# excluded (homopoly):', excl_homo],
	             ['OUTPUT_VARS:', TOTAL_OUTPUT_VARIANTS]]

	# print stats
	for n in OUT_PRINT:
		print(n[0].ljust(OUT_STRING_SIZE) + str(n[1]))

if __name__ == '__main__':
	main()
