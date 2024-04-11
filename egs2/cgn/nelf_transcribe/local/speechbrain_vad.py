import os, sys
import torch
from speechbrain.inference.VAD import VAD

## Apply VAD with pre-trained model
## Ref: https://huggingface.co/speechbrain/vad-crdnn-libriparty

filename = sys.argv[1]
audio_file = sys.argv[2]
model_dir = sys.argv[3]
out_dir = sys.argv[4]
ngpu = int(sys.argv[5])

## SETTINGS ##
# vad
large_chunk_size = 60  # load 60s audio at a time
small_chunk_size = 10  # process 10s audio parts in parallel
overlap_small_chunk = True  # 50% overlap between small chunks

activation_th = 0.20   # def=0.50, post>thresh = starting a speech segment 
deactivation_th = 0.20 # def=0.25, post<thresh = ending a speech segment

en_activation_th = 0.50 # energy>thresh = starting a speech segment
de_activation_th = 0.20 # energy<thresh = ending a speech segment

len_th=0.25    # merge if segment length < len_th
close_th=0.25  # merge if segment distance < close_th

min_energy_distance = 0.50  # initial minimum pause between energy segments
max_energy_distance = 5.0   # for sanity, keeps the algorithm simple

max_segment_len = 25  # maximum length of segment in sec
min_segment_len = 0.5

# post-processing
final_min_dist = 1.0 # if distance smaller than this, collide the boundaries

extend_value_begin = 0.2 # extend start boundary of segment (if not < final_min_dist)
extend_value_end = 0.0  # extend end boundary of segment
##################

if ngpu > 0:
    device = "cuda"
else:
    device = "cpu"

print('[Segment] Running VAD on device=%s' % device)

assert os.path.exists(os.path.join(model_dir, 'model.ckpt')), "VAD Model does not exist in %s" % model_dir

# Load VAD model
VAD = VAD.from_hparams(source=model_dir, savedir=out_dir, run_opts={"device":device})
#Download with: source="speechbrain/vad-crdnn-libriparty"

# Compute speech segment boundaries
#boundaries = VAD.get_speech_segments(audio_file, large_chunk_size=60, small_chunk_size=10, overlap_small_chunk=True, activation_th=0.20, deactivation_th=0.20)

print('[Segment] Applying CRDNN VAD model')

# Compute speech vs non speech probs
prob_chunks = VAD.get_speech_prob_file(audio_file, large_chunk_size=large_chunk_size, small_chunk_size=small_chunk_size, overlap_small_chunk=overlap_small_chunk)

# Apply a threshold to get candidate speech segments
prob_th = VAD.apply_threshold(prob_chunks, activation_th=activation_th, deactivation_th=deactivation_th).float()

# Compute the boundaries of speech segments
boundaries = VAD.get_boundaries(prob_th, output_value="seconds")

# Merge short segments
boundaries = VAD.merge_close_segments(boundaries, close_th=close_th)

# Remove short segments
boundaries = VAD.remove_short_segments(boundaries, len_th=len_th)

n_segs = boundaries.shape[0]

# Split up too long segments based on Energy VAD
iter_k = 0
while True:
    #print('boundaries: ', boundaries)
    segment_lengths = boundaries[:, 1] - boundaries[:, 0]

    print('[Segment] VAD - num segments = %i / average segment = %i s / longest segment = %i s' % (
        segment_lengths.shape[0],
        torch.mean(segment_lengths).cpu().item(),
        torch.max(segment_lengths).cpu().item(),
        )
    )

    too_long_segs = segment_lengths > max_segment_len

    if not too_long_segs.any():
        # all segments have appropriate length
        break
    iter_k += 1

    print('[Segment] Applying Energy VAD to split too long segments - Iter %i' % iter_k)

    too_long_boundaries = boundaries[too_long_segs]

    # Apply energy VAD
    energy_boundaries = VAD.energy_VAD(audio_file, too_long_boundaries, activation_th=en_activation_th, deactivation_th=de_activation_th)

    # Find segments that are spaced somewhat far apart (= longest silence between them)
    energy_segment_distance = energy_boundaries[1:, 0] - energy_boundaries[:-1, 1]
    energy_boundaries = torch.cat((energy_boundaries[0, :].unsqueeze(0), energy_boundaries[1:, :][(energy_segment_distance > min_energy_distance) & (energy_segment_distance < max_energy_distance)]), dim=0)

    segment_centers = too_long_boundaries[:, 1] - (segment_lengths[too_long_segs] / 2)

    # Find non-speech frame that is closest to center of segment
    insert_indices = torch.searchsorted(energy_boundaries[:, 0].contiguous(), segment_centers.contiguous(), side="right")
    insert_indices = torch.clamp(insert_indices, 0, energy_boundaries.shape[0]-1)

    # Split in two halves
    pt1_clone = too_long_boundaries.clone()
    pt1_clone[:, 1] = energy_boundaries[insert_indices][:, 0]

    pt2_clone = too_long_boundaries.clone()
    pt2_clone[:, 0] = energy_boundaries[insert_indices][:, 1]

    res = torch.stack((pt1_clone, pt2_clone), dim=1).view(2*too_long_boundaries.shape[0], 2)

    # Add new shorter segments to previous short segments
    boundaries = torch.cat([pt1_clone, pt2_clone, boundaries[~too_long_segs]], dim=0)
    boundaries, _ = torch.sort(boundaries, dim=0)

    boundaries = VAD.remove_short_segments(boundaries, len_th=min_segment_len)

    if n_segs == boundaries.shape[0]:
        #print('Reducing distance')
        min_energy_distance = min_energy_distance / 2
        en_activation_th = en_activation_th / 2
    n_segs = boundaries.shape[0]

boundaries = VAD.remove_short_segments(boundaries, len_th=min_segment_len)

print('[Segment] Finished VAD')
print('[Segment] Creating segments file for input file')

# Write segments
with open(os.path.join(out_dir, 'segments'), 'w') as td:
    #cnt_seg = 1
    last_end = 0.0
    total_segs = boundaries.shape[0]
    for i in range(total_segs):
        if boundaries[i, 0] - last_end < final_min_dist:
            begin_value = last_end
        else:
            begin_value = max(last_end, boundaries[i, 0] - extend_value_begin)
        if i == total_segs - 1:  # final segment
            end_value = boundaries[i, 1]
        else:
            end_value = min(boundaries[i, 1] + extend_value_end, boundaries[i+1, 0])
        td.write("%s-%.7d-%.7d %s %.2f %.2f\n" % (filename, (100.0*begin_value), (100.0*end_value), filename, begin_value, end_value))
        #cnt_seg += 1
        last_end = end_value

