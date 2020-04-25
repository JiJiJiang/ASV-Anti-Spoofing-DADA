#!/bin/bash
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com
# Created on: 20190225

# Extract features
./scripts/extract_features.sh ASV17
./scripts/extract_features.sh BTAS16-PA

# Make features
for data_set in ASV17 BTAS16-PA; do
    cat data/${data_set}/features/STFT-LPS/STFT-LPS_train.ark data/${data_set}/features/STFT-LPS/STFT-LPS_dev.ark > data/${data_set}/features/STFT-LPS/STFT-LPS_train-dev.ark
done
mkdir -p data/ASV17+BTAS16-PA/features/STFT-LPS
cat data/ASV17/features/STFT-LPS/STFT-LPS_train-dev.ark data/BTAS16-PA/features/STFT-LPS/STFT-LPS_train-dev.ark > data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark

echo "Finish making features."
