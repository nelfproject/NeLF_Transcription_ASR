import os, sys

filename = sys.argv[1]
infile = sys.argv[2]
scratchdir = sys.argv[3]

datadir = os.path.join(scratchdir, 'data')

with open(os.path.join(datadir, 'wav.scp'), 'w') as td:
    td.write('%s %s\n' % (filename, infile))

with open(os.path.join(datadir, 'segments'), 'r') as td, open(os.path.join(datadir, 'utt2spk'), 'w') as uttfile, open(os.path.join(datadir, 'text'), 'w') as textfile, open(os.path.join(datadir, 'text.lc.verbatim'), 'w') as verbfile, open(os.path.join(datadir, 'text.lc.subtitle'), 'w') as subsfile:
    line = td.readline()
    while line:
        seg, fn, st, et = line.rstrip().split(' ')
        textfile.write('%s X\n' % seg)
        uttfile.write('%s %s-spk\n' % (seg, seg))
        verbfile.write('%s X\n' % seg)
        subsfile.write('%s X\n' % seg)
        line = td.readline()
    
