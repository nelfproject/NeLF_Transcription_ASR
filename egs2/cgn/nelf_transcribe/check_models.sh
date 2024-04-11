#!/usr/bin/env bash

version=$1
annotation=$2
vadtype=$3

err_out=false

if [ $version = "v1" ]; then
  if [ $annotation = "verbatim" ]; then
    modeldir=./model/ASR_verbatim_v1/
  else
    modeldir=./model/ASR_subtitles_v1/
  fi
elif [ $version = "v2" ]; then
  modeldir=./model/ASR_subtitles_v2/
fi

if [ $modeltype = "wordtimings" ]; then
  modeldir=./model/ASR_wordtimings_v1/
fi

if [ ! -f ${modeldir}/model.pth ]; then
  echo "Model is not present in ${modeldir}. Clone/copy the repository from https://huggingface.co/nelfproject/ and place in ${modeldir}"
  err_out=true
fi

if [ $vadtype = "CRDNN" ]; then
  vadmodel=./model/vad/
  if [ ! -f ${vadmodel}/model.ckpt ]; then
    echo "VAD model is not present in ${vadmodel}. Copy/clone the model files from https://huggingface.co/speechbrain/vad-crdnn-libriparty/ and put in ${vadmodel}"
    err_out=true
  fi
fi

if [ "$err_out" = true ]; then
  exit 1;
else
  echo "All required models are found."
fi
