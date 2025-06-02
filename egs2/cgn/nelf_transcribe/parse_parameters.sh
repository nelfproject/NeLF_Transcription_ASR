#!/usr/bin/env bash

# defaults
VERSION="v2"
VADMODEL="rVAD"
MODELTYPE="transcribe"
DEVICE="GPU"
ANNOTATION="both"
FORMATTING="clean"
TIMESTAMPS="timestamps"

# parse input parameters
while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        # model version: v1 or v2 (latest)
        --version)
        shift
        VERSION="$1"
        ;;
        # vad type: rvad or cdrnn
        --vad)
        shift
        VADMODEL="$1"
        ;;
        # model type: wordtiming or transcription (=best)
        --type)
        shift
        MODELTYPE="$1"
        ;;
        # cpu or gpu
        --device)
        shift
        DEVICE="$1"
        ;;
        # annotation to generate: subtitle, verbatim or both
        --annot)
        shift
        ANNOTATION="$1"
        ;;
        # cleanup settings: include tags or not
        --formatting)
        shift
        FORMATTING="$1"
        ;;
        # output settings: include vad timestamps or not
        --timestamps)
        shift
        TIMESTAMPS="$1"
        ;;
        *)
        # Unknown options or input/output stuff
        ;;
    esac
    shift
done

if [ $DEVICE = "GPU" ]; then
  ngpu=1
else
  ngpu=0
fi


echo "    VERSION = ${VERSION}"
echo "    VADMODEL = ${VADMODEL}"
echo "    MODELTYPE = ${MODELTYPE}"
echo "    DEVICE = ${DEVICE}"
echo "    ANNOTATION = ${ANNOTATION}"
echo "    FORMATTING = ${FORMATTING}"
echo "    TIMESTAMPS = ${TIMESTAMPS}"
echo "    ngpu = ${ngpu}"

