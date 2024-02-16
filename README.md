# pacbio-tools
Various scripts and workflows for processing PacBio data. Written in Python3.

## filter_homopolymer_indels.py
Remove homopolymer indels from a VCF.

## filter_vcf.py
Remove variants that fail local realignment from a VCF.

align.so will require Cython recompilation in order for this to function elsewhere:

```
Cython -2 align.pyx
python3 setup.py build_ext --inplace
```

## get_zmw_coverage.py
Computes ZMW coverage depth from a BAM file.

```
python3 get_zmw_coverage.py -i input.bam -m CCS -o out/
```
