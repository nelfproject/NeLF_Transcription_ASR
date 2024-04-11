import os, sys

datadir = sys.argv[1]

with open(os.path.join(datadir, 'segments'), 'r') as pd, open(os.path.join(datadir, 'segments_extended'), 'w') as td:
    line = pd.readline()
    prev_seg, prev_fn, prev_st, prev_et = None, None, None, None
    while line:
        seg, fn, st, et = line.rstrip().split(' ')
        if fn == prev_fn:
            if prev_et is not None:
                prev_et_float = float(prev_et)
                st_float = float(st)
                if st_float > prev_et_float:
                    mid_point = round((st_float + prev_et_float)/2, 2)
                    prev_et = str(mid_point)
                    st = str(mid_point)
        else:
            st = "0.00"
        if prev_seg is not None:
            td.write(' '.join([prev_seg, prev_fn, prev_st, prev_et])+'\n')
            prev_seg, prev_fn, prev_st, prev_et = seg, fn, st, et
        else:
            prev_seg, prev_fn, prev_st, prev_et = seg, fn, "0.00", et
        line = pd.readline()
    td.write(' '.join([seg, fn, st, et])+'\n')

