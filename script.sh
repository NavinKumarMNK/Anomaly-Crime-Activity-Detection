#!/bin/bash

if command -v conda >/dev/null 2>&1 ; then
    echo "Miniconda is already installed on this system."
else
    # Download the latest version of Miniconda for Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

    # Install Miniconda
    bash ~/miniconda.sh -b -p $HOME/miniconda

    # Add Miniconda to PATH
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

    echo "Miniconda has been installed."
fi

conda create --name ray
conda activate ray
conda install python=3.10.8
pip install -U "ray[default]"
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

