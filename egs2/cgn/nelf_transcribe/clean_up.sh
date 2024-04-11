#!/usr/bin/env bash

scratchdir=$1
annotation=$2
formatting=$3
timestamps=$4

mkdir -p $scratchdir/result

if [ "${annotation}" = verbatim ]; then
    python local/cleanup.py $scratchdir/decode/text $scratchdir/data/segments $scratchdir/result/transcription_verbatim $formatting $timestamps
elif [ "${annotation}" = subtitle ]; then
    python local/cleanup.py $scratchdir/decode/subs $scratchdir/data/segments $scratchdir/result/transcription_subtitle $formatting $timestamps

elif [ "${annotation}" = both ]; then
    python local/cleanup.py $scratchdir/decode/text $scratchdir/data/segments $scratchdir/result/transcription_verbatim $formatting $timestamps
    python local/cleanup.py $scratchdir/decode/subs $scratchdir/data/segments $scratchdir/result/transcription_subtitle $formatting $timestamps
fi

