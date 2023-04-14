#!/bin/bash
git clone https://github.com/alibaba/federatedscope.git
cd federatedscope
git checkout 0f172fa5fd6763a633b2401262e9648ea4ae1ff3
pip install --upgrade pip
pip install numpy scikit-learn scipy pandas opencv-python-headless pytest
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
pip install fvcore iopath
pip install wandb tensorboard tensorboardX pympler
pip install grpcio grpcio-tools protobuf==3.19.4 setuptools==61.2.0
cp ../federatedscope.patch ./federatedscope.patch
git apply --whitespace=fix federatedscope.patch
pip install -e .
rm -rf ~/.cache/pip
cd ..