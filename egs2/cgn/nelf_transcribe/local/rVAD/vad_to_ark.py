import os, sys
import numpy as np
import kaldiio

filename = sys.argv[1]
vadfile = sys.argv[2]
outark = sys.argv[3]
outscp = sys.argv[4]

vad_dict = {}
with open(vadfile, 'r') as td:
    lines = td.readlines()
vad = np.array([int(line.strip()) for line in lines], dtype=np.int32)
vad_dict[filename] = vad

kaldiio.save_ark(outark, vad_dict, scp=outscp)

