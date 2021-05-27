import os
import re
import sys

# python cleanup_svs_from_multiple_tools.py /folder/of/vcfs/

# collapse all types into DEL/INS
# pass only
# remove alt seq
# remove all <40bp
# ouput simplified metadata
# -- blank out id, ref, GT, other info

SVTYPE_DICT = {'DUP':        'INS',
               'DUP:INT':    'INS',
               'DUP:TANDEM': 'INS',
               'INS':        'INS',
               'INS:NOVEL':  'INS',
               'CNV':        'INS',
               'DEL':        'DEL'}

DIR = sys.argv[1]
if DIR[-1] != '/':
	DIR += '/'

VCF = [DIR + n for n in os.listdir(DIR) if n[-4:] == '.vcf']

for fn in VCF:
	f = open(fn,'r')
	f2 = open(fn+'.simple','w')
	for line in f:
		if line[0] == '#':
			f2.write(line)
		else:
			splt = line.strip().split('\t')
			splt[2] = '.'						# blank out id
			splt[3] = 'N'						# blank out ref
			if splt[6].upper() != 'PASS':		# pass only
				continue
			myInfo  = splt[7]
			re_end  = re.findall(r";END=.*?(?=;)",';'+myInfo+';')[0]
			re_type = re.findall(r";SVTYPE=.*?(?=;)",';'+myInfo+';')[0]
			if re_type[8:] not in SVTYPE_DICT:	# remove invs, etc
				continue
			re_len  = re.findall(r";SVLEN=.*?(?=;)",';'+myInfo+';')[0]
			if abs(int(re_len[7:])) < 40:			# >=40 only
				continue
			myType = SVTYPE_DICT[re_type[8:]]
			splt[4] = '<'+myType+'>'
			splt[7] = splt[7].replace(re_type[1:],'SVTYPE='+myType)
			splt[8] = 'GT'
			splt[9] = './.'
			f2.write('\t'.join(splt)+'\n')
	f2.close()
	f.close()
