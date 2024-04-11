#!/usr/bin/env python

import os, sys
import re
import datetime

text = sys.argv[1]
segments = sys.argv[2]
outfile = sys.argv[3]

# tags or clean
text_fmt = sys.argv[4]
# timestamps or notimestamps
output_fmt = sys.argv[5]


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def replace_bracket_all_tags(match):
    if match.group(1).startswith('lang'):
        return ''
    elif match.group(1).startswith('task'):
        return ''
    elif match.group(1).endswith('timestamps'):
        return ''
    elif match.group(1).startswith('*'):
        # tag, drop completely
        return ''
    elif is_float(match.group(1)):
        # timing, keep as is
        return match.group(0)
    elif match.group(1) in ['xxx','ggg']:
        # non-speech sounds, drop
        return ''
    elif match.group(1) == 'spk':
        return ''
    elif match.group(1).startswith('spk-'):
        return ''
    else:
        # punctuation, drop brackets
        return match.group(1)

def replace_bracket_tokens(match):
    if match.group(1).startswith('lang'):
        return ''
    elif match.group(1).startswith('task'):
        return ''
    elif match.group(1).endswith('timestamps'): 
        return ''
    else:
        # dont filter
        return match.group(0)

seg2text = dict()
with open(text, 'r') as pd:
    line = pd.readline()
    while line:
        seg = line.rstrip().split(' ')[0]
        txt = ' '.join(line.rstrip().split(' ')[1:])
        if text_fmt == "tags":
            # keep all tags, except decoding prompts
            txt_clean = re.sub(r'\<(.*?)\>', replace_bracket_tokens, txt)
        elif text_fmt == "clean":
            # remove all tags, except timing
            txt_clean = re.sub(r'\<(.*?)\>', replace_bracket_all_tags, txt)
        else:
            raise NotImplementedError('unknown formatting type %s, should be clean/tags' % text_fmt)
        
        # remove multiple spaces
        txt_clean = re.sub(r'\s+', ' ', txt_clean)

        # Upper case letter after .?!
        txt_out = re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), txt_clean)

        seg2text[seg] = txt_out
        line = pd.readline()

prev_txt = ''
with open(segments, 'r') as pd, open(outfile, 'w', encoding="utf-8") as td:
    line = pd.readline()
    while line:
        seg, fn, st, et = line.rstrip().split(' ')
        txt_out = seg2text[seg]
        if output_fmt == "timestamps":
            st, et = float(st), float(et)
            st = '{:02d}:{:02d}:{:05.2f}'.format(int(st // 3600), int(st % 3600 // 60), st % 60)
            et = '{:02d}:{:02d}:{:05.2f}'.format(int(et // 3600), int(et % 3600 // 60), et % 60)
            td.write('[ %s --> %s ] %s\n' % (st, et, txt_out))
        elif output_fmt == "notimestamps":
            if prev_txt.endswith(('.','?','!')) or prev_txt.endswith(('.>','?>','!>')):
                txt_out = txt_out.capitalize()
            td.write('%s ' % txt_out)
        else:
            raise NotImplementedError('unknown timestamps mode type %s, should be timestamps/notimestamps' % output_fmt)
        prev_txt = txt_out
        line = pd.readline()


