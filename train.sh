#!/bin/bash

# the Kaldi path
. ./path.sh

seed=1

### baseline: LightCNN

# ASV17_train_dev #
python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17/labels/train_dev.txt --features=data/ASV17/features/STFT-LPS/STFT-LPS_train-dev.ark --model=LightCNN_9Layers --seed=$seed
# BTAS16-PA_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/BTAS16-PA/labels/train_dev.txt --features=data/BTAS16-PA/features/STFT-LPS/STFT-LPS_train-dev.ark --model=LightCNN_9Layers --seed=$seed
# ASV17+BTAS16-PA_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev.txt --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=LightCNN_9Layers --seed=$seed

### baseline: LightCNN_DAT

# ASV17+BTAS16-PA_train_dev DAT #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='ASV17' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=LightCNN_9Layers_DAT --label_type='DAT' --seed=$seed
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='BTAS16-PA' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=LightCNN_9Layers_DAT --label_type='DAT' --seed=$seed

### proposed: LightCNN_DADA

# ASV17+BTAS16-PA_train_dev DADA #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='ASV17' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=LightCNN_9Layers_DADA --label_type='DADA' --seed=$seed
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='BTAS16-PA' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=LightCNN_9Layers_DADA --label_type='DADA' --seed=$seed




### baseline: CGCNN

# ASV17_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17/labels/train_dev.txt --features=data/ASV17/features/STFT-LPS/STFT-LPS_train-dev.ark --model=CGCNN_10Layers --seed=$seed
# BTAS16-PA_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/BTAS16-PA/labels/train_dev.txt --features=data/BTAS16-PA/features/STFT-LPS/STFT-LPS_train-dev.ark --model=CGCNN_10Layers --seed=$seed
# ASV17+BTAS16-PA_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev.txt --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=CGCNN_10Layers --seed=$seed

### baseline: CGCNN_DAT

# ASV17+BTAS16-PA_train_dev DAT #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='ASV17' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=CGCNN_10Layers_DAT --label_type='DAT' --seed=$seed
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='BTAS16-PA' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=CGCNN_10Layers_DAT --label_type='DAT' --seed=$seed

### proposed: CGCNN_DADA

# ASV17+BTAS16-PA_train_dev DADA #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='ASV17' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=CGCNN_10Layers_DADA --label_type='DADA' --seed=$seed
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='BTAS16-PA' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=CGCNN_10Layers_DADA --label_type='DADA' --seed=$seed




### baseline: ResNet_10Layers

# ASV17_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17/labels/train_dev.txt --features=data/ASV17/features/STFT-LPS/STFT-LPS_train-dev.ark --model=ResNet_10Layers --seed=$seed
# BTAS16-PA_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/BTAS16-PA/labels/train_dev.txt --features=data/BTAS16-PA/features/STFT-LPS/STFT-LPS_train-dev.ark --model=ResNet_10Layers --seed=$seed
# ASV17+BTAS16-PA_train_dev #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev.txt --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=ResNet_10Layers --seed=$seed

### baseline: ResNet_10Layers_DAT

# ASV17+BTAS16-PA_train_dev DAT #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='ASV17' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=ResNet_10Layers_DAT --label_type='DAT' --seed=$seed
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='BTAS16-PA' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=ResNet_10Layers_DAT --label_type='DAT' --seed=$seed

### proposed: ResNet_10Layers_DADA

# ASV17+BTAS16-PA_train_dev DADA #
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='ASV17' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=ResNet_10Layers_DADA --label_type='DADA' --seed=$seed
#python3 -u scripts/run.py train --config='config/train.yaml' --labels=data/ASV17+BTAS16-PA/labels/ASV17-train-dev_BTAS16-PA-train-dev_DAT-DADA.txt --outdomain='BTAS16-PA' --features=data/ASV17+BTAS16-PA/features/STFT-LPS/STFT-LPS_ASV17-train-dev_BTAS16-PA-train-dev.ark --model=ResNet_10Layers_DADA --label_type='DADA' --seed=$seed

