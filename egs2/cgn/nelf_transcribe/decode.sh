#!/usr/bin/env bash

scratchdir=$1
ngpu=$2
version=$3
modeltype=$4
annotation=$5

if [ $version = "v1" ]; then
  if [ $annotation = "verbatim" ]; then
    modeldir=./model/ASR_verbatim_v1
    mode=asr
  else
    modeldir=./model/ASR_subtitles_v1
    mode=subs
  fi
elif [ $version = "v2" ]; then
  modeldir=./model/ASR_subtitles_v2
  mode=subs
fi

if [ $modeltype = "wordtimings" ]; then
  modeldir=./model/ASR_wordtimings_v1
  mode=owsm
fi

. ./cmd.sh
. ./path.sh

if [ ${ngpu} -gt 0 ]; then
  echo "[DECODE] Decoding on GPU"
  _cmd="${cuda_cmd}"
  inference_nj=1
  batch_size=1
else
  echo "[DECODE] Decoding on CPU"
  _cmd="${decode_cmd}"
  inference_nj=16
  batch_size=1
fi
_ngpu=${ngpu}

decodedir=$scratchdir/decode
logdir=$decodedir/log
datadir=$scratchdir/feats

mkdir -p $decodedir
mkdir -p $logdir

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

if [ "${mode}" = asr ]; then
    inference_tool="espnet2.bin.asr_inference"
    _opts="--config ${modeldir}/decode.yaml --asr_train_config ${modeldir}/train.yaml --asr_model_file ${modeldir}/model.pth "
elif [ "${mode}" = subs ]; then
    inference_tool="espnet2.bin.subtitling_inference_chained"
    _opts="--st_train_config ${modeldir}/train.yaml --st_model_file ${modeldir}/model.pth "
    if [ $annotation == "verbatim" ]; then
	_opts+="--config ${modeldir}/decode_verbatim.yaml "
    else
	_opts+="--config ${modeldir}/decode.yaml "
    fi
elif [ "${mode}" = owsm ]; then
    inference_tool="espnet2.bin.s2t_subtitle_inference"
    _opts="--st_train_config ${modeldir}/train.yaml --st_model_file ${modeldir}/model.pth "
    if [ $annotation == "verbatim" ]; then
      _opts+="--config ${modeldir}/decode_wordtimings_verbatim.yaml "
    else
      _opts+="--config ${modeldir}/decode_wordtimings.yaml "
    fi
else
    echo "Invalid decoding mode: mode = ${mode}"
    exit 1;
fi

_feats_type="$(<${datadir}/feats_type)"
if [ "${_feats_type}" = raw ]; then
    _scp=wav.scp
    if [[ "${audio_format}" == *ark* ]]; then
        _type=kaldi_ark
    else
        _type=sound
    fi
else
    _scp=feats.scp
    _type=kaldi_ark
fi

key_file=${datadir}/${_scp}
_nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${logdir}/keys.${n}.scp"
done
            
utils/split_scp.pl "${key_file}" ${split_scps}

echo "[Decode] Decoding started... log: '${logdir}/inference.*.log'"

# If number of threads is a problem, use the following line for inference
#${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${logdir}"/inference.JOB.log \
#OMP_NUM_THREADS=${_nj},3,1 ${_cmd} --num_threads 4 --max-jobs-run 1 JOB=1:"${_nj}" "${logdir}"/inference.JOB.log \

${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${logdir}"/inference.JOB.log \
                  python -m ${inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${datadir}/${_scp},speech,${_type}" \
                    --key_file "${logdir}"/keys.JOB.scp \
                    --output_dir "${logdir}"/output.JOB \
                    ${_opts} ${inference_args}

echo "[Decode] Decode done. Collecting output."
for f in token token_int score text subs score_subs token_subs token_int_subs; do
    if [ -f "${logdir}/output.1/1best_recog/${f}" ]; then
        for i in $(seq "${_nj}"); do
            cat "${logdir}/output.${i}/1best_recog/${f}"
        done | LC_ALL=C sort -k1 >"${decodedir}/${f}"
    fi
done

echo "[Decode] Done."
