# Next Level Flemish Speech Recognition
This is a separate branch of the ESPnet toolkit with new neural networks for ASR on Flemish Dutch. This repository contains all the code to use the ASR models from the NeLF Project, which aims to attain Next Level Flemish Speech Processing with state-of-the-art Flemish Dutch speech recognition models.

This codebase is required to use the pre-trained ASR models from https://huggingface.co/nelfproject.

# Installation
To use this repository, first setup your environment. An environment.yml file has been included in tools/ to build an environment with working version dependencies. 

Using conda:    
      conda env create -f environment.yml

(see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). 

Next, follow the ESPnet installation procedure to install all the tools. The instalation is detailed in https://espnet.github.io/espnet/installation.html . Make sure to include step 1 (kaldi installation). In step 2, clone this code branch instead of the master espnet branch and use your previously built environment.

# Usage
After installing all the tools, packages and dependencies, you can start using our ASR models to transcribe Flemish audio!

Go to egs2/cgn/nelf_transcribe and follow the instructions.

# Information
For more information about Flemish Speech recognition in the NeLF project, visit our website: https://nelfproject.be 

We also offer a webservice to transcribe audio for you, if you request access.

**version: 11/04/2024 (Jakob Poncelet)
