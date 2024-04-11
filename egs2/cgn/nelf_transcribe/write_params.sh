#!/usr/bin/env bash

scratchdir=$1
filename=$2
version=$3
vadmodel=$4
modeltype=$5
annotation=$6
formatting=$7
timestamps=$8

outfile=$scratchdir/result/parameters

cat << EOF > $outfile
File = ${filename}

PARAMETERS
Model Version: ${version}
VAD Model: ${vadmodel}
Model Type: ${modeltype}
Annotations: ${annotation}
Output tags: ${formatting}
Output timestamps: ${timestamps}

EOF
