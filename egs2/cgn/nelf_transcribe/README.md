# NeLF ASR Transcription Pipeline

This directory contains all scripts to transcribe and annotate speech with NeLF models.

# Installation
Make sure that this branch of ESPnet is correctly installed, as well as Kaldi (in tools/kaldi).

Download and clone/copy the desired models and all files from https://huggingface.co/nelfproject/ to the correct directory in ./model/

To use the CRDNN Voice activity detection model, download the files from https://huggingface.co/speechbrain/vad-crdnn-libriparty/ to ./model/vad/

# Usage
To transcribe an audio file, use the following command:

     pipeline.sh <inputfile> <settings>

For example, to transcribe the file data/example.mp3 using the default (recommended) settings, run the command:
  
     pipeline.sh data/example.mp3 --version v2 --vad CRDNN --type transcribe --annot both --formatting clean --timestamps timestamps --device GPU


The possible settings are:
     --version      v1 | v2
        Model Version: Select which version you want to use
        Default: v2 (latest)

     --vad          CRDNN | rVAD
        Voice Activity Detection: Model to use for Voice Activity Detection
        Default: CRDNN

     --type         transcribe | wordtimings
        Model Type: Transcribe the input recordings with an ASR model (best transcriptions) or generate word-level timings (still experimental!) with a transcription+timing model
        Default: transcribe

     --annot        verbatim | subtitle | both
        Annotation Type: Generate a verbatim transcription, a subtitle transcription, or both
        Default: both

     --formatting   clean | tags
        Formatting: The transcription should contain all tags (e.g. <*d>, <.>) or should be cleaned up text
        Default: clean

     --timestamps   notimestamps | timestamps
        Timestamps: Include start/end time stamps for speech sentences or generate one large body of text without timestamps
        Default: timestamps

     --device       CPU | GPU
        Device: Decode on GPU (fast) or CPU (slow). Processing time on GPU is estimated at around 0.25-0.50 RTF depending on the chosen configuration, and requires ~5GB GPU memory.
        Default: GPU

# Contact
For any help or questions, feel free to contact me.
Jakob Poncelet: jakob.poncelet@kuleuven.be

**version**: 11/04/2024 (Jakob Poncelet)
