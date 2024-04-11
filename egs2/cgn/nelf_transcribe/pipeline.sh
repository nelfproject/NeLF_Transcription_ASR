#!/usr/bin/env bash

echo "JOB: pipeline.sh $@"

inputfile=$1

# default settings
VERSION="v2"
VADMODEL="CRDNN"
MODELTYPE="transcribe"
DEVICE="GPU"
ANNOTATION="both"
FORMATTING="clean"
TIMESTAMPS="timestamps"

# parse parameters
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

. ./path.sh
. ./cmd.sh

curdir=`pwd`

# 0. Extract file identifiers for input/output
filename=$(basename "$inputfile")
extension="${filename##*.}"
fileid=$(basename "$filename" .$extension)
scratchdir=./scratch/$(fileid)_$(date +"%y_%m_%d_%H_%M_%S_%N")
mkdir -p $scratchdir || fatalerror "Unable to create temporary working directory $scratchdir"
echo "Working directory: $scratchdir"

# 1. Check if all models are correctly downloaded
echo "### Check required models ###"
. check_models.sh $VADMODEL $VERSION $MODELTYPE

# 2. Convert to correct wav format
echo "### Extracting 16kHz/mono wav from input ###"
convertedfile=${scratchdir}/${fileid}.wav
ffmpeg -hide_banner -i "$inputfile" -vn -sample_fmt s16 -ac 1 -ar 16000 ${convertedfile} || fatalerror "Failure calling ffmpeg"
inputfile=${convertedfile}

# 3. Segment input file
echo "### Segmenting input file ###"
. segment.sh $fileid $inputfile $scratchdir $ngpu $VADMODEL

# 4. Prepare data files
echo "### Preparing data files ###"
. prepare.sh $fileid $inputfile $scratchdir

# 5. Extract filterbanks
echo "### Extracting filterbanks ###"
. extract_feats.sh $scratchdir

# 6. Decode input file
echo '### Decoding ###'
. decode.sh $scratchdir $ngpu $VERSION $MODELTYPE $ANNOTATION

# 7. Clean output
echo '### Cleaning output ###'
. clean_up.sh $scratchdir $ANNOTATION $FORMATTING $TIMESTAMPS

# 8. Write parameters
echo '### Saving parameter settings ###'
. write_params.sh $scratchdir $fileid $VERSION $VADMODEL $MODELTYPE $ANNOTATION $FORMATTING $TIMESTAMPS

echo 'DONE: Saved output at $scratchdir/result'

