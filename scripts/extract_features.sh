#!/bin/bash
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com
# Created on: 20190225

data_set=$1 # ASV17 or BTAS16-PA
echo "Dataset:" ${data_set}
for mode in train dev eval; do
    feats_type=STFT-LPS
    mkdir -p data/${data_set}/features/${feats_type}
    python3 scripts/extract_STFT-LPS.py data/${data_set}/flists/${mode}.scp data/${data_set}/features/${feats_type}/${feats_type}_${mode}.ark
done

