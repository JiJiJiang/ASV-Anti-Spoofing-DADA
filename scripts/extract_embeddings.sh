#!/bin/bash
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com
# Created on: 20190304

. ./path.sh

model_path=$1
feature_type=STFT-LPS

for dataset in ASV17 BTAS16-PA; do
    for subset in train-dev; do
        python3 scripts/run.py extract --model_path=$model_path/model.th --features=data/${dataset}/features/${feature_type}/${feature_type}_${subset}.ark --output_arkfile=${dataset}_embeddings_${subset}.ark
        copy-vector ark:$model_path/${dataset}_embeddings_${subset}.ark ark,t:$model_path/${dataset}_embeddings_${subset}.ark.txt
    done
done
