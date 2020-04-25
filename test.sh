#!/bin/bash

. ./path.sh

model_path=experiments/LightCNN_9Layers/
#model_path=experiments/CGCNN_10Layers/
#model_path=experiments/ResNet_10Layers/
feature_type=STFT-LPS
label_type=label # label, DAT, or DADA

for dataset in ASV17 BTAS16-PA; do
    python3 scripts/run.py score --model_path=$model_path/model.th --features=data/${dataset}/features/${feature_type}/${feature_type}_eval.ark --label_type=${label_type} --dataset=${dataset} --output=predictions_${dataset}_eval.txt --eval_out=evaluation_${dataset}_eval.txt
done
