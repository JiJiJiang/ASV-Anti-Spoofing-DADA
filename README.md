# ASV-Anti-Spoofing-DADA

A Pytorch implementation of the Dual-Adversarial Domain Adaptation (DADA) approach for replay spoofing detection in automatic speaker verification (ASV).

Paper has been submitted to [INTERSPEECH 2020](http://www.interspeech2020.org/).

Please get your running environment ready firstly, e.g., [Pytorch 1.1+](http://pytorch.org/), [Kaldi](https://github.com/kaldi-asr/kaldi), etc. **Specially, do not forget to update `path.sh` for Kaldi.**

### Table of Contents
- <a href='#Datasets'>Datasets</a>
- <a href='#Feature_Extraction'>Feature_Extraction</a>
- <a href='#Models'>Models</a>
- <a href='#Training'>Training</a>
- <a href='#Testing'>Testing</a>
- <a href='#Extracting_Embeddings'>Extracting_Embeddings</a>
- <a href='#Results'>Results</a>
- <a href='#Citation'>Citation</a>

## Datasets

We use two real-case replay datasets, whese replay attacks are replayed and recorded using real devices, rather than artificially simulated (e.g., [ASVspoof 2019](https://datashare.is.ed.ac.uk/handle/10283/3336)):

* [ASVspoof 2017](https://datashare.is.ed.ac.uk/handle/10283/3055) uses a variety of replay configurations (acoustic environment, recording and playback devices). It focuses on `in the wild` scenes.
* BTAS 2016 is based on [AVspoof](https://www.idiap.ch/dataset/avspoof) dataset. This dataset contains both PA and LA attacks, but only the PA portion (denoted as BTAS16-PA) is used in our experiments.


## Feature_Extraction

257-dimensional `log power spectrograms (LPS)` are extracted as front-end features by computing 512-point Short-Time Fourier Transform (STFT) every 10 ms with a window size of 25 ms.
Here we use The [librosa](https://github.com/librosa/librosa) toolkit (code entry: `scripts/make_features.sh`).

Also, the [Kaldi](https://github.com/kaldi-asr/kaldi) toolkit is employed to
apply sliding-window cmvn per utterance.

## Models

Three typical anti-spoofing models are evaluated:
* `LCNN`: Light CNN is the winner of the ASVspoof2017 challenge [(paper)](https://pdfs.semanticscholar.org/a2b4/c396dc1064fb90bb5455525733733c761a7f.pdf). We use the adapted version in our previous work [(paper)](https://pdfs.semanticscholar.org/72a8/fd18652d55aa2c9e99bc629233fcfb6fe61a.pdf), which applies to variable lengths of input features.
* `ResNet10`: The ResNet variations used in ASVspoof 2019 achieved great performance in the PA subtask. We use the 10-layer ResNet for it is comparable with LCNN in parameter size.
* `CGCNN`: Context-Gate CNN was our main proposal in ASVspoof 2019 [(paper)](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2170.pdf). Gated linear unit (GLU) activations are used to replace the MFM activations in LCNN. Except for that, CGCNN shares a similar structure with LCNN. 

All model definitions can be seen in `scripts/models.py`.

## Training

The framework here uses a combination of [google-fire](https://github.com/google/python-fire) and `yaml` parsing to enable a convenient interface.
By default one needs to pass a `*.yaml` configuration file into any of the command scripts.
However, parameters of the `yaml` files (seen in `config/train.yaml`) can also be overwritten.

e.g., to train a LCNN model using training data in the ASV17 dataset:

```
python3 -u scripts/run.py train --config=config/train.yaml \
                                --labels=data/ASV17/labels/train_dev.txt \
                                --features=data/ASV17/features/STFT-LPS/STFT-LPS_train-dev.ark \
                                --model=LightCNN_9Layers \
                                --label_type=label \
                                --seed=1
```

All training commands are listed in `train.sh`.

## Testing

After training, we get the model path. To test the model, the command is:

```
python3 scripts/run.py score --model_path=MODEL_PATH/model.th \
                             --features=FEATURES.ark \
                             --label_type=label/DAT/DADA \
                             --dataset=ASV17/BTAS16-PA \
                             --output=OUTPUT.txt \
                             --eval_out=EVALUATION_RESULT.txt
```

Just run `test.sh` using correct testing parameters.
Evaluation scripts are directly taken from the baseline of the ASVspoof2019 challenge, seen [here](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip).

## Extracting_Embeddings

We can extract deep embeddings for t-SNE visualization. The command is:

```
python3 scripts/run.py extract --model_path=MODEL_PATH/model.th \
                               --features=FEATURES.ark \
                               --output_arkfile=OUTPUT.ark
```

Please refer to `scripts/extract_embeddings.sh`.

## Results

We pool the training set and the development set as the actual training data, 10% of which are further divided asthe validation set for model selection.
All models are tested on both evaluation sets.

The results are as follows:

![Results.png](https://s1.ax1x.com/2020/04/25/Jyk01O.png)

## Citation

If you use our models, please cite the following papers:
```
@article{wang2019cross,
  title={Cross-domain replay spoofing attack detection using domain adversarial training},
  author={Wang, Hongji and Dinkel, Heinrich and Wang, Shuai and Qian, Yanmin and Yu, Kai},
  journal={Proc. Interspeech 2019},
  pages={2938--2942},
  year={2019}
}

@article{yang2019sjtu,
  title={The SJTU Robust Anti-spoofing System for the ASVspoof 2019 Challenge},
  author={Yang, Yexin and Wang, Hongji and Dinkel, Heinrich and Chen, Zhengyang and Wang, Shuai and Qian, Yanmin and Yu, Kai},
  journal={Proc. Interspeech 2019},
  pages={1038--1042},
  year={2019}
}
```
