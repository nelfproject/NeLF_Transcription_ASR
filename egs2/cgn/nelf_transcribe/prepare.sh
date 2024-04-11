#!/usr/bin/env bash

filename=$1
inputfile=$2
scratchdir=$3
curdir=`pwd`

mkdir -p $scratchdir/data

echo "[Prepare] Preparing data files"
cp $scratchdir/VAD/segments $scratchdir/data/segments

python local/prepare_data.py $filename $inputfile $scratchdir

utils/utt2spk_to_spk2utt.pl $scratchdir/data/utt2spk > $scratchdir/data/spk2utt
