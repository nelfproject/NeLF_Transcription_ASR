#!/usr/bin/env bash

scratchdir=$1
nj=8

datadir=$scratchdir/data
featsdir=$scratchdir/feats

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

. ./cmd.sh

# 0. Copy data
echo "[Extract] Copying data to $featsdir"
utils/copy_data_dir.sh $datadir $featsdir >/dev/null

#utils/fix_data_dir.sh $featsdir

# 1. Extract fbanks
echo "[Extract] Extracting filterbanks"
_nj=$(min "${nj}" "$(<"${featsdir}/utt2spk" wc -l)")

#. ./path-kaldi.sh >/dev/null 2>&1
steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${featsdir}" || exit 1;
#. ./path-espnet2.sh

# 2. Validate fbank extraction
echo "[Extract] Validating extracted features"
utils/fix_data_dir.sh --utt_extra_files "text.lc.verbatim text.lc.subtitle" "${featsdir}"

# 3. Derive the the frame length and feature dimension
echo "[Extract] Computing feature length and dimension"
scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
	"${featsdir}/feats.scp" "${featsdir}/feats_shape"

# 4. Write feats_dim
head -n 1 "${featsdir}/feats_shape" | awk '{ print $2 }' \
	| cut -d, -f2 > ${featsdir}/feats_dim

# 5. Write feats_type
echo "fbank_pitch" > "${featsdir}/feats_type"
