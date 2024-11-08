#!/bin/bash

real_esrgan_dir=$(dirname $(dirname ${BASH_SOURCE[0]}))
model_dir="$real_esrgan_dir"/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P $model_dir
