import os
import re
import bisect
import argparse

from common.ref_func_corgi import index_ref, read_ref_py3, exists_and_is_nonZero, rm, IND_2_LEXICO, LEXICO_2_IND, HG38_SIZES

"""
Edit distance computation.
Copied from https://bitbucket.org/marcelm/sqt/src/af255d54a21815cb9a3e0b279b431a320d4626bd/sqt/_helpers.pyx
"""
from common.align import edit_distance, edit_distance_affine_gap

REF_CHAR  = 'MX=D'
READ_CHAR = 'MX=IS'

def get_sam_lines_from_bam(in_bam, samtools, temp_sam=None, specific_ref=None):
	if temp_sam == None:
		temp_sam = in_bam+'.tmpsam'
	if specific_ref == None:
		cmd = samtools + ' view ' + in_bam + ' > ' + temp_sam
	elif specific_ref in HG38_SIZES:
		my_region = specific_ref + ':1-' + str(HG38_SIZES[specific_ref])
		cmd = samtools + ' view ' + in_bam + ' ' + my_region + ' > ' + temp_sam
	else:
		'WHAT REF DID YOU FEED ME???', specific_ref
		exit(1)
	os.system(cmd)
	if exists_and_is_nonZero(temp_sam):
		f = open(temp_sam, 'r')
		fr = f.read()
		f.close()
		rm(temp_sam)
		return [n for n in fr.split('\n') if len(n)]
	else:
		return None

def main(raw_args=None):
	parser = argparse.ArgumentParser(description='filter_vcf.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
	parser.add_argument('-r',  type=str, required=True,  default=argparse.SUPPRESS, metavar='<ref.fa>',     help="* Path to reference fasta")
	parser.add_argument('-o',  type=str, required=True,  default=argparse.SUPPRESS, metavar='<output.vcf>', help="* Path to output vcf")
	parser.add_argument('-cb', type=str, required=False, default='',                metavar='<ccs.bam>',    help="Path to CCS/Hifi bam")
	parser.add_argument('-cv', type=str, required=False, default='',                metavar='<ccs.vcf>',    help="Path to CCS/Hifi vcf")
	parser.add_argument('-sb', type=str, required=False, default='',                metavar='<clr.bam>',    help="Path to CLR/subreads bam")
	parser.add_argument('-sv', type=str, required=False, default='',                metavar='<clr.vcf>',    help="Path to CLR/subreads vcf")
	parser.add_argument('-m',  type=int, required=False, default=10,                metavar='<MAPQ>',       help="Toss reads below this MAPQ")
	parser.add_argument('-s',  type=str, required=False, default='samtools',        metavar='<samtools>',   help="Path to samtools exe")
	args = parser.parse_args(raw_args)

	IN_REF  = args.r
	OUT_VCF = args.o

	CCS_BAM = args.cb
	CCS_VCF = args.cv
	SUB_BAM = args.sb
	SUB_VCF = args.sv

	SAMTOOLS = args.s

	MAX_MULTIMAP   = 2
	REALIGN_BUFFER = 20
	TOO_MUCH_AMBIG = 0.5

	MIN_DP_CCS = 2	# from input vcf
	MIN_DP_SUB = 5	# from input vcf
	MIN_SUPPORT_CCS = 2	# from realignment
	MIN_SUPPORT_SUB = 5 # from realignment

	MAPQ_THRESHOLD = args.m

	HAVE_CCS = len(CCS_BAM) or len(CCS_VCF)
	HAVE_SUB = len(SUB_BAM) or len(SUB_VCF)

	if HAVE_CCS == False and HAVE_SUB == False:
		print('Must include either CCS/Hifi or CLR/subreads bam + vcf.')
		exit(1)
	if HAVE_CCS and (len(CCS_BAM) == 0 or len(CCS_VCF) == 0):
		print('You must input both bam and vcf input (CCS/Hifi).')
		exit(1)
	if HAVE_SUB and (len(SUB_BAM) == 0 or len(SUB_VCF) == 0):
		print('You must input both bam and vcf input (CLR/subreads).')
		exit(1)

	if HAVE_CCS:
		if exists_and_is_nonZero(CCS_BAM) == False:
			print('CCS/Hifi bam not found.')
			exit(1)
		if exists_and_is_nonZero(CCS_VCF) == False:
			print('CCS/Hifi vcf not found.')
			exit(1)
	if HAVE_SUB:
		if exists_and_is_nonZero(SUB_BAM) == False:
			print('CLR/subreads bam not found.')
			exit(1)
		if exists_and_is_nonZero(SUB_VCF) == False:
			print('CLR/subreads vcf not found.')
			exit(1)
	if exists_and_is_nonZero(IN_REF) == False:
		print('Reference fasta not found.')
		exit(1)

	vcf_dir  = '/'.join(OUT_VCF.split('/')[:-1])
	if len(vcf_dir) == 0:
		vcf_dir = './'
	else:
		vcf_dir += '/'
	temp_sam = vcf_dir + 'temp.sam'

	# stats
	TOTAL_INPUT_VARIANTS = 0
	TOTAL_OUTPUT_VARIANTS = 0
	VARS_FILTERED_DP = 0
	VARS_FILTERED_REALIGN_ALT_SUPPORT = 0
	VARS_FILTERED_AMBIG = 0

	#
	# read reference sequence
	#
	ref_inds = index_ref(IN_REF)
	ri_rev   = {ref_inds[n][0]:n for n in range(len(ref_inds))}
	ref_list  = {}

	#
	# read vcfs
	#
	var_list_ccs = {}
	var_list_sub = {}
	header_ccs   = ''
	header_sub   = ''
	print('reading input vcf...')
	if HAVE_CCS:
		f = open(CCS_VCF, 'r')
		for line in f:
			if line[0] == '#':
				header_ccs += line
				continue
			TOTAL_INPUT_VARIANTS += 1
			splt = line.strip().split('\t')
			if splt[0] not in ref_list:
				ref_list[splt[0]] = True
			my_info = splt[7] + ';'
			my_dp   = int(re.findall(r"DP=.*?(?=;)", my_info)[0][3:])
			if my_dp < MIN_DP_CCS:
				VARS_FILTERED_DP += 1
				continue
			if ',' in splt[3] or ',' in splt[4]:	# skip sites with multiple alleles for now
				continue
			if splt[0] not in var_list_ccs:
				var_list_ccs[splt[0]] = []
			# chr, pos, ref, alt, dp, whole_line
			var_list_ccs[splt[0]].append([splt[0], int(splt[1]), splt[3], splt[4], my_dp, line])
		f.close()
	if HAVE_SUB:
		f = open(SUB_VCF, 'r')
		for line in f:
			if line[0] == '#':
				header_sub += line
				continue
			TOTAL_INPUT_VARIANTS += 1
			splt = line.strip().split('\t')
			if splt[0] not in ref_list:
				ref_list[splt[0]] = True
			my_info = splt[7] + ';'
			my_dp   = int(re.findall(r"DP=.*?(?=;)", my_info)[0][3:])
			if my_dp < MIN_DP_SUB:
				continue
			if ',' in splt[3] or ',' in splt[4]:	# skip sites with multiple alleles for now
				continue
			if splt[0] not in var_list_sub:
				var_list_sub[splt[0]] = []
			# chr, pos, ref, alt, dp, whole_line
			var_list_sub[splt[0]].append([splt[0], int(splt[1]), splt[3], splt[4], my_dp, line])
		f.close()
	sorted_ref_list = [IND_2_LEXICO[n] for n in sorted([LEXICO_2_IND[k] for k in ref_list.keys()])]

	#
	# heavy lifting!
	#
	f_out_vcf = open(OUT_VCF, 'w')
	f_out_vcf.write(header_ccs)

	for my_ref in sorted_ref_list:

		#
		# read bams
		#
		print('reading bam alignments for ' + my_ref + '...')
		read_list_ccs = []
		read_list_sub = []
		rnm_count_ccs = {}
		rnm_count_sub = {}
		if HAVE_CCS:
			sam_dat = get_sam_lines_from_bam(CCS_BAM, SAMTOOLS, temp_sam=temp_sam, specific_ref=my_ref)
			for line in sam_dat:
				if line[0] == '@':
					continue
				splt = line.strip().split('\t')
				if int(splt[4]) < MAPQ_THRESHOLD:	# low mapq
					continue
				orientation = 'FWD'
				if int(splt[1])&16:
					orientation = 'REV'
				# rname, chr, pos, cigar, seq, qual, orientation
				read_list_ccs.append([splt[0], splt[2], int(splt[3]), splt[5], splt[9], splt[10], orientation, line])
				if splt[0] not in rnm_count_ccs:
					rnm_count_ccs[splt[0]] = 0
				rnm_count_ccs[splt[0]] += 1
		if HAVE_SUB:
			sam_dat = get_sam_lines_from_bam(SUB_BAM, SAMTOOLS, temp_sam=temp_sam, specific_ref=my_ref)
			for line in sam_dat:
				if line[0] == '@':
					continue
				splt = line.strip().split('\t')
				if int(splt[4]) < MAPQ_THRESHOLD:	# low mapq
					continue
				orientation = 'FWD'
				if int(splt[1])&16:
					orientation = 'REV'
				# rname, chr, pos, cigar, seq, qual, orientation
				read_list_sub.append([splt[0], splt[2], int(splt[3]), splt[5], splt[9], splt[10], orientation, line])
				if splt[0] not in rnm_count_sub:
					rnm_count_sub[splt[0]] = 0
				rnm_count_sub[splt[0]] += 1
	
		#
		#
		#
		my_ri   = ri_rev[my_ref]
		ref_seq = read_ref_py3(IN_REF, ref_inds[my_ri])

		my_vars_ccs = []
		if my_ref in var_list_ccs:
			print('processing ' + str(len(var_list_ccs[my_ref])) + ' variants from CCS on ' + my_ref + '...')
			my_vars_ccs = sorted(var_list_ccs[my_ref])
		my_vars_sub = []
		if my_ref in var_list_sub:
			print('processing ' + str(len(var_list_ccs[my_ref])) + ' variants from SUBREADS on ' + my_ref + '...')
			my_vars_sub = sorted(var_list_sub[my_ref])
		blist_ccs = [n[1] for n in my_vars_ccs]
		blist_sub = [n[1] for n in my_vars_sub]

		PREVIOUSLY_PRINTED = 0
		PRINT_EVERY        = 100

		if HAVE_CCS:
			#my_reads_ccs = [n for n in read_list_ccs if (n[1] == my_ref and rnm_count_ccs[n[0]] <= MAX_MULTIMAP)]
			my_reads_ccs = read_list_ccs
			####var_matrix = []
			var_support_dict = {}
			del_list   = []
			for read_i in range(len(my_reads_ccs)):

				my_span   = (my_reads_ccs[read_i][2], my_reads_ccs[read_i][2] + len(my_reads_ccs[read_i][4]))
				var_ind_s = bisect.bisect(blist_ccs, my_span[0])
				var_ind_e = bisect.bisect(blist_ccs, my_span[1])
				if var_ind_s == var_ind_e:	# we span no variants
					continue
				var_ind_s   = max([0, var_ind_s-1])
				var_ind_e   = min([len(my_vars_ccs), var_ind_e+1])
				my_var_inds = list(range(var_ind_s, var_ind_e))

				if read_i+1 >= PREVIOUSLY_PRINTED + PRINT_EVERY:
					print(my_ref+',', 'read:', read_i+1, '/', len(my_reads_ccs), '- spans', len(my_var_inds), 'variant sites')
					PREVIOUSLY_PRINTED = read_i+1

				cigar   = my_reads_ccs[read_i][3]
				letters = re.split(r"\d+",cigar)[1:]
				numbers = [int(n) for n in re.findall(r"\d+",cigar)]
				[adj, radj] = [0,0]
				refPos_to_readPos = {}
				for i in range(len(letters)):
					if letters[i] in 'MX=':
						for j in range(numbers[i]):
							refPos_to_readPos[my_reads_ccs[read_i][2]+adj+j] = radj+j
					if letters[i] in REF_CHAR:
						adj += numbers[i]
					if letters[i] in READ_CHAR:
						radj += numbers[i]

				####var_hits   = [None for n in my_vars_ccs]
				####var_bQual  = [None for n in my_vars_ccs]
				ambigCount = 0
				callCount  = 0
				#for i in range(len(my_vars_ccs)):
				for i in my_var_inds:
					if i > 0:
						lb = min([my_vars_ccs[i][1] - my_vars_ccs[i-1][1], REALIGN_BUFFER])
					else: 
						lb = REALIGN_BUFFER
					if i < len(my_vars_ccs)-1:
						ub = min([my_vars_ccs[i+1][1] - my_vars_ccs[i][1], REALIGN_BUFFER])
					else: 
						ub = REALIGN_BUFFER
					alt_buff   = len(my_vars_ccs[i][3])-1
					ref_buff   = len(my_vars_ccs[i][2])-1
					#if alt_buff == 0 and ref_buff == 0:	# this code is for skipping everything except indels
					#	continue
					rpos       = my_vars_ccs[i][1]
					subseq_ref = str(ref_seq[rpos-1-lb:rpos+ub+alt_buff])
					subseq_alt = str(ref_seq[rpos-1-lb:rpos-1]) + my_vars_ccs[i][3] + str(ref_seq[rpos+ref_buff:rpos+ub+ref_buff])
					#print(i, my_vars_ccs[i], (lb, ub))
					#print(subseq_ref, 'ref')
					#print(subseq_alt, 'alt')
					if rpos in refPos_to_readPos:
						readpos = refPos_to_readPos[rpos]
						if readpos+1+REALIGN_BUFFER >= len(my_reads_ccs[read_i][4])-1 or readpos-REALIGN_BUFFER < 0:
							continue
						subseq_read = my_reads_ccs[read_i][4][readpos-lb:readpos+1+ub+alt_buff]
						subseq_qual = my_reads_ccs[read_i][5][readpos-lb:readpos+1+ub+alt_buff]
						#print( subseq_read, 'read')
						#print( subseq_qual)
						myQual = min([ord(n) for n in subseq_qual[lb-1:lb+2]])
						#myQual = 100
						####var_bQual[i] = myQual
						distance_ref = edit_distance(subseq_read, subseq_ref)
						distance_alt = edit_distance(subseq_read, subseq_alt)
						#print('read vs. ref:', distance_ref)
						#print('read vs. alt:', distance_alt)
						#print('')
						if i not in var_support_dict:
							var_support_dict[i] = []
						if distance_ref < distance_alt:
							####var_hits[i] = -myQual
							var_support_dict[i].append(-myQual)
							callCount  += 1
						elif distance_ref > distance_alt:
							####var_hits[i] = myQual
							var_support_dict[i].append(myQual)
							callCount  += 1
						else:
							####var_hits[i] = 0.
							var_support_dict[i].append(0.)
							ambigCount += 1
				####var_matrix.append([n for n in var_hits])
			#
			#for i in range(len(var_matrix[0])):
			for i in sorted(var_support_dict.keys()):
				#read_support = [var_matrix[j][i] for j in range(len(var_matrix)) if var_matrix[j][i] != None]
				read_support = var_support_dict[i]
				alt_support  = len([n for n in read_support if n > 0])
				ref_support  = len([n for n in read_support if n < 0])
				unambig_frac = (alt_support + ref_support)/float(len(read_support))
				# why did we even bother?
				if alt_support < MIN_SUPPORT_CCS:
					VARS_FILTERED_REALIGN_ALT_SUPPORT += 1
					continue
				# too much ambiguous sites
				if unambig_frac <= 1. - TOO_MUCH_AMBIG:
					VARS_FILTERED_AMBIG += 1
					continue
				# ok!
				#print(i, alt_support, '/', ref_support, unambig_frac, my_vars_ccs[i], read_support)
				f_out_vcf.write(my_vars_ccs[i][5])
				TOTAL_OUTPUT_VARIANTS += 1

			#break

	f_out_vcf.close()

	print('----------------')
	print('INPUT_VARS:   ', TOTAL_INPUT_VARIANTS)
	print('FILT_COVERAGE:', VARS_FILTERED_DP)
	print('FILT_REALIGN: ', VARS_FILTERED_REALIGN_ALT_SUPPORT)
	print('FILT_AMBIG:   ', VARS_FILTERED_AMBIG)
	print('OUTPUT_VARS:  ', TOTAL_OUTPUT_VARIANTS)
	print('----------------')

if __name__ == '__main__':
	main()
