from __future__ import print_function

import sys
import os

#
#	Misc hg38 variables
#
LEXICO_2_IND = {'chr1':1, 'chr2':2, 'chr3':3, 'chr10':10, 'chr11':11, 'chr12':12, 'chr19':19, 'chr20':20,
				'chr4':4, 'chr5':5, 'chr6':6, 'chr13':13, 'chr14':14, 'chr15':15, 'chr21':21, 'chr22':22,
				'chr7':7, 'chr8':8, 'chr9':9, 'chr16':16, 'chr17':17, 'chr18':18, 'chrX' :23, 'chrY' :24, 'chrM' :25,
				'chr1_p':26, 'chr1_q':27, 'chr2_p':28, 'chr2_q':29, 'chr3_p':30, 'chr3_q':31,
                'chr4_p':32, 'chr4_q':33, 'chr5_p':34, 'chr5_q':35, 'chr6_p':36, 'chr6_q':37,
                'chr7_p':38, 'chr7_q':39, 'chr8_p':40, 'chr8_q':41, 'chr9_p':42, 'chr9_q':43,
                'chr10_p':44, 'chr10_q':45, 'chr11_p':46, 'chr11_q':47, 'chr12_p':48, 'chr12_q':49,
                'chr13_p':50, 'chr13_q':51, 'chr14_p':52, 'chr14_q':53, 'chr15_p':54, 'chr15_q':55,
                'chr16_p':56, 'chr16_q':57, 'chr17_p':58, 'chr17_q':59, 'chr18_p':60, 'chr18_q':61,
                'chr19_p':62, 'chr19_q':63, 'chr20_p':64, 'chr20_q':65, 'chr21_p':66, 'chr21_q':67,
                'chr22_p':68, 'chr22_q':69, 'chrX_p':70, 'chrX_q':71, 'chrY_p':72, 'chrY_q':73}

HG38_SIZES = {'chr1':248956422,  'chr2':242193529,  'chr3':198295559,  'chr4':190214555,  'chr5':181538259,
              'chr6':170805979,  'chr7':159345973,  'chr8':145138636,  'chr9':138394717,  'chr10':133797422,
              'chr11':135086622, 'chr12':133275309, 'chr13':114364328, 'chr14':107043718, 'chr15':101991189,
              'chr16':90338345,  'chr17':83257441,  'chr18':80373285,  'chr19':58617616,  'chr20':64444167,
              'chr21':46709983,  'chr22':50818468,  'chrX':156040895,  'chrY':57227415,   'chrM':16569}

# chm13 t2t reference v2.0
T2T11_SIZES = {'chr1':248387328,  'chr2':242696752,  'chr3':201105948,  'chr4':193574945,  'chr5':182045439,
               'chr6':172126628,  'chr7':160567428,  'chr8':146259331,  'chr9':150617247,  'chr10':134758134,
               'chr11':135127769, 'chr12':133324548, 'chr13':113566686, 'chr14':101161492, 'chr15':99753195,
               'chr16':96330374,  'chr17':84276897,  'chr18':80542538,  'chr19':61707364,  'chr20':66210255,
               'chr21':45090682,  'chr22':51324926,  'chrX':154259566,  'chrY':62460029,   'chrM':16569}

TELOGATOR_SIZES = {'chr1_p':500000, 'chr1_q':500000, 'chr2_p':500000, 'chr2_q':500000, 'chr3_p':500000, 'chr3_q':500000,
                   'chr4_p':500000, 'chr4_q':500000, 'chr5_p':500000, 'chr5_q':500000, 'chr6_p':500000, 'chr6_q':500000,
                   'chr7_p':500000, 'chr7_q':500000, 'chr8_p':500000, 'chr8_q':500000, 'chr9_p':500000, 'chr9_q':500000,
                   'chr10_p':500000, 'chr10_q':500000, 'chr11_p':500000, 'chr11_q':500000, 'chr12_p':500000, 'chr12_q':500000,
                   'chr13_p':500000, 'chr13_q':500000, 'chr14_p':500000, 'chr14_q':500000, 'chr15_p':500000, 'chr15_q':500000,
                   'chr16_p':500000, 'chr16_q':500000, 'chr17_p':500000, 'chr17_q':500000, 'chr18_p':500000, 'chr18_q':500000,
                   'chr19_p':500000, 'chr19_q':500000, 'chr20_p':500000, 'chr20_q':500000, 'chr21_p':500000, 'chr21_q':500000,
                   'chr22_p':500000, 'chr22_q':500000, 'chrX_p':500000, 'chrX_q':500000, 'chrY_p':500000, 'chrY_q':500000}

IND_2_LEXICO = {LEXICO_2_IND[k]:k for k in LEXICO_2_IND.keys()}

#
#	Index reference fasta, if necessary
#
def index_ref(ref_path):
	fn = None
	if os.path.isfile(ref_path+'i'):
		print('found index '+ref_path+'i')
		fn = ref_path+'i'
	if os.path.isfile(ref_path+'.fai'):
		print('found index '+ref_path+'.fai')
		fn = ref_path+'.fai'

	ref_inds = []
	if fn != None:
		fai = open(fn,'r')
		for line in fai:
			splt = line[:-1].split('\t')
			seqLen = int(splt[1])
			offset = int(splt[2])
			lineLn = int(splt[3])
			nLines = seqLen//lineLn
			if seqLen%lineLn != 0:
				nLines += 1
			ref_inds.append((splt[0],offset,offset+seqLen+nLines,seqLen))
		fai.close()
		return ref_inds

	sys.stdout.write('index not found, creating one... ')
	sys.stdout.flush()
	refFile = open(ref_path,'r')
	prevR   = None
	prevP   = None
	seqLen  = 0
	while 1:
		data = refFile.readline()
		if not data:
			ref_inds.append( (prevR, prevP, refFile.tell()-len(data), seqLen) )
			break
		if data[0] == '>':
			if prevP != None:
				ref_inds.append( (prevR, prevP, refFile.tell()-len(data), seqLen) )
			seqLen = 0
			prevP  = refFile.tell()
			prevR  = data[1:-1]
		else:
			seqLen += len(data)-1
	refFile.close()

	print('{0:.3f} (sec)'.format(time.time()-tt))
	return ref_inds

#
#	Get sequence string from indexed reference
#
def read_ref_py3(ref_path, ref_inds_i):
	ref_file = open(ref_path,'r')
	ref_file.seek(ref_inds_i[1])
	my_dat = ''.join(ref_file.read(int(ref_inds_i[2])-int(ref_inds_i[1])).split('\n'))
	ref_file.close()
	return my_dat

#
#	Various other functions
#
RC_DICT = {'A':'T','C':'G','G':'C','T':'A','N':'N'}
def RC(s):
	return ''.join([RC_DICT[n] for n in s[::-1]])

def exists_and_is_nonZero(fn):
	if os.path.isfile(fn):
		if os.path.getsize(fn) > 0:
			return True
	return False

def makedir(d):
	if not os.path.isdir(d):
		os.system('mkdir '+d)

def rm(fn):
	if os.path.isfile(fn):
		os.system('rm '+fn)
