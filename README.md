# Next Level Flemish Speech Recognition
This is a separate branch of the ESPnet toolkit with new neural networks for ASR on Flemish Dutch. This repository contains all the code to use the ASR models from the NeLF Project, which aims to attain Next Level Flemish Speech Processing with state-of-the-art Flemish Dutch speech recognition models.

This codebase is required to use the pre-trained ASR models from https://huggingface.co/nelfproject.

# Installation 
The installation procedure closely follows the espnet installation (https://espnet.github.io/espnet/installation.html), but it is simplified by recreating a working environment.
Please follow the steps below.

1) Install Kaldi (external installation: https://kaldi-asr.org/doc/install.html).
2) Install Anaconda (external installation: https://docs.anaconda.com/anaconda/install/).
3) Clone this repository:
   
     git clone https://github.com/nelfproject/NeLF_Transcription_ASR
   
4) Put the compiled Kaldi under tools. The kaldi-root is the path to directory 'kaldi' after installation.
   
     cd tools
   
     ln -s KALDI_ROOT .
   
5) Create the conda environment from the environment.yml file. 
    (see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). 

     cd tools
   
     conda env create -f environment.yml
   
6) Activate the conda environment.
   
     conda activate nelf
   
7) Add the conda environment to the espnet installation.
   The CONDA_ROOT is the place where your Anaconda is installed, e.g. ./anaconda3.
   You can also get it automatically from the CONDA_PREFIX variable which is set by Anaconda after installing and activating an environment.

   cd tools
   
   CONDA_ROOT=${CONDA_PREFIX}/../..

   ./setup_anaconda.sh ${CONDA_ROOT} nelf 3.11

Now you have finished the installation for this espnet branch.

You don't have to run the Makefile, the environment.yml contains all the packages.

# Usage
After installation, you can start using our ASR models to transcribe Flemish audio!

Go to egs2/cgn/nelf_transcribe and follow the instructions.

# Information
For more information about Flemish Speech recognition in the NeLF project, visit our website: https://nelfproject.be 

We also offer a webservice to transcribe audio for you, if you request access.

**version: 11/04/2024 (Jakob Poncelet)

# Research paper
The models, training data and experiments are described in the following paper: https://arxiv.org/abs/2502.03212 
If you use our code and models, please consider citing it.

```bibtex
@article{poncelet2024,
    author = "Poncelet, Jakob and Van hamme, Hugo",
    title = "Leveraging Broadcast Media Subtitle Transcripts for Automatic Speech Recognition and Subtitling",
    year={2024},
    journal={arXiv preprint arXiv:2502.03212},
    url = {https://arxiv.org/abs/2502.03212}
}

