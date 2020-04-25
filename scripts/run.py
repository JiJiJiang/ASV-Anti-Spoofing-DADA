# coding=utf-8
#!/usr/bin/env python3
import datetime
import torch
from pprint import pformat
import models
import fire
import loss
import logging
import pandas as pd
import kaldi_io
import yaml
import os
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pathlib import Path
import tableprint as tp
import sklearn.preprocessing as pre
import torchnet as tnt
from torch.utils.data import Dataset
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from tabulate import tabulate
import math

class ListDataset(Dataset):
    """Dataset wrapping List.

    Each sample will be retrieved by indexing List along the first dimension.

    Arguments:
        *lists (List): List that have the same size of the first dimension.
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(a_list) for a_list in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(a_list[index] for a_list in self.lists)

    def __len__(self):
        return len(self.lists[0])


def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    outstr = "copy-feats ark:{} ark:- |".format(feat)
    if cmvn:
        outstr += "apply-cmvn-sliding --center --norm-vars --cmn-window=300 --max-warnings=0 ark:- ark:- |"
        #outstr += "compute-cmvn-stats ark:- ark:- | apply-cmvn --norm-vars ark:- ark:{} ark:- |".format(feat)
    if delta:
        outstr += "add-deltas ark:- ark:- |"
    if splice and splice > 0:
        outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
            splice, splice)
    return outstr


def runepoch(dataloader1, dataloader2=None, model=None, criterion=None, target_label_name='label', optimizer=None, dotrain=True, epoch=1):
    model = model.train() if dotrain else model.eval()
    # By default use average pooling
    loss_meter1 = tnt.meter.AverageValueMeter()
    loss_meter2 = tnt.meter.AverageValueMeter()
    loss_meter3 = tnt.meter.AverageValueMeter()
    acc_meter1 = tnt.meter.ClassErrorMeter(accuracy=True)
    acc_meter2 = tnt.meter.ClassErrorMeter(accuracy=True)
    acc_meter3 = tnt.meter.ClassErrorMeter(accuracy=True)
    # DAT or DADA parameters
    sigma = 0.01
    alpha = 2/(1+math.exp(-sigma*epoch))-1

    run_indomain = True
    if dataloader2 == None: 
        dataloader2 = dataloader1
        run_indomain = False
    # unified framework for baseline, DAT and DADA
    with torch.set_grad_enabled(dotrain):
        for i, ((outdomain_features, outdomain_targets),(indomain_features, indomain_targets)) in enumerate(zip(dataloader1, dataloader2), 1):
            # baseline (one domain only): outdomain=indomain
            features = outdomain_features.float().to(device)
            targets = outdomain_targets.to(device)
            
            if target_label_name == 'label':
                outputs = model(features)
                loss = criterion(outputs, targets)
            elif target_label_name == 'DAT':
                outputs = model(features)
                out1, out2 = outputs
                loss_task1 = criterion(out1, targets[:, 0])
                loss_task2 = criterion(out2, targets[:, 1]) * alpha
                loss = loss_task1 + loss_task2
            else: # 'DADA'
                # bonafide_outdomain
                loss_task1 = loss_task2 = loss_task3 = torch.tensor(0.0).to(device)
                bonafide_outdomain_index = (targets[:, 0] == 0)
                if sum(bonafide_outdomain_index).item() > 0:
                    bonafide_features = features[bonafide_outdomain_index]
                    bonafide_targets = targets[bonafide_outdomain_index]
                    out1_1, out2 = model(bonafide_features, domain='bonafide_outdomain')
                    loss_task1 = criterion(out1_1, bonafide_targets[:,0])
                    loss_task2 = criterion(out2, bonafide_targets[:,1]) * alpha
                # spoof_outdomain
                spoof_outdomain_index = (targets[:, 0] == 1)
                if sum(spoof_outdomain_index).item() > 0:
                    spoof_features = features[spoof_outdomain_index]
                    spoof_targets = targets[spoof_outdomain_index]
                    out1_2, out3 = model(spoof_features, domain='spoof_outdomain')
                    loss_task1 += criterion(out1_2, spoof_targets[:,0])
                    loss_task3 = criterion(out3, spoof_targets[:,1]) * alpha
                loss = loss_task1 + loss_task2 + loss_task3

            # loss, acc
            if target_label_name == 'label':
                loss_meter1.add(loss.item())
                acc_meter1.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
            elif target_label_name == 'DAT':
                loss_meter1.add(loss_task1.item())
                acc_meter1.add(out1.cpu().detach().numpy(), targets[:, 0].cpu().numpy())
                loss_meter2.add(loss_task2.item())
                acc_meter2.add(out2.cpu().detach().numpy(), targets[:, 1].cpu().numpy())
            else: # 'DADA'
                loss_meter1.add(loss_task1.item())
                if sum(bonafide_outdomain_index).item() > 0:
                    loss_meter2.add(loss_task2.item())
                    acc_meter1.add(out1_1.cpu().detach().numpy(), bonafide_targets[:, 0].cpu().numpy())
                    acc_meter2.add(out2.cpu().detach().numpy(), bonafide_targets[:, 1].cpu().numpy())
                if sum(spoof_outdomain_index).item() > 0:
                    loss_meter3.add(loss_task3.item())
                    acc_meter1.add(out1_2.cpu().detach().numpy(), spoof_targets[:, 0].cpu().numpy())
                    acc_meter3.add(out3.cpu().detach().numpy(), spoof_targets[:, 1].cpu().numpy())

            loss_indomain = torch.tensor(0.0).to(device)
            if run_indomain:
                ##### indomain #####
                features = indomain_features.float().to(device)
                targets = indomain_targets.to(device)
                if target_label_name == 'DAT':
                    out2 = model(features, domain='indomain')
                    loss_task2 = criterion(out2, targets[:, 1]) * alpha
                    loss_indomain = loss_task2
                else: #'DADA'
                    out1, out2, out3 = model(features, domain='indomain')
                    criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
                    loss_task2 = sum(out1[:, 0] * criterion2(out2, targets[:, 1])) / len(targets) * alpha
                    loss_task3 = sum(out1[:, 1] * criterion2(out3, targets[:, 1])) / len(targets) * alpha
                    loss_indomain = loss_task2 + loss_task3
                # loss, acc
                loss_meter2.add(loss_task2.item())
                acc_meter2.add(out2.cpu().detach().numpy(), targets[:, 1].cpu().numpy())
                if target_label_name == 'DADA':
                    loss_meter3.add(loss_task3.item())
                    acc_meter3.add(out3.cpu().detach().numpy(), targets[:, 1].cpu().numpy())
                
            if dotrain:
                # updata the model 
                optimizer.zero_grad()
                loss = loss_indomain + loss
                loss.backward()
                optimizer.step()

    if target_label_name == 'label':
        return loss_meter1.value()[0], acc_meter1.value()[0]
    elif target_label_name == 'DAT':
        return (loss_meter1.value()[0], loss_meter2.value()[0]), (acc_meter1.value()[0], acc_meter2.value()[0])
    else: # 'DADA'
        return (loss_meter1.value()[0], loss_meter2.value()[0], loss_meter3.value()[0]), (acc_meter1.value()[0], acc_meter2.value()[0], acc_meter3.value()[0])


def genlogger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
        level=logging.DEBUG,
        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read)
    # values from config file are all possible params
    help_str = "Valid Parameters are:\n"
    help_str += "\n".join(list(yaml_config.keys()))
    # passed kwargs will override yaml config
    for key in kwargs.keys():
        assert key in yaml_config, "Parameter {} invalid!\n".format(
            key) + help_str
    return dict(yaml_config, **kwargs)


### repeat
def collate_fn(data_batches):
    """collate_fn

    Helper function for torch.utils.data.Dataloader

    :param data_batches: iterateable
    """
    data_batches.sort(key=lambda x: len(x[0]), reverse=True)

    def merge_seq(dataseq, dim=0):
        lengths = [seq.shape for seq in dataseq]
        # Assuming duration is given in the first dimension of each sequence
        maxlengths = tuple(np.max(lengths, axis=dim))
        maxlengths = (min(1500, maxlengths[0]), maxlengths[1])
        # For the case that the lengths are 2dimensional
        lengths = np.array(lengths)[:, dim]
        padded = np.zeros((len(dataseq),) + maxlengths)
        for i, seq in enumerate(dataseq):
            tile_seq = np.tile(seq, (maxlengths[0]//len(seq)+1, 1))
            end = lengths[i]
            padded[i] = tile_seq[:maxlengths[0]]
        return padded, lengths
    features, targets = zip(*data_batches)
    features_seq, feature_lengths = merge_seq(features)
    return torch.from_numpy(features_seq), torch.LongTensor(targets)


def create_dataloader_train_cv(
        feature_string,
        train_label_dict,
        transform=None,
        target_label_name='label',
        outdomain_label=0,
        **kwargs):
    """create_dataloader_train_cv

    :param feature_string: kaldi feature pipline string e.g., copy-feats ark:file.ark ark:- |
    :param train_label_dict: Mappings from each kaldi ark file to label
    :param transform: Feature transformation, usually scaler.transform
    :param **kwargs: Other parameters
    """
    train_percentage = kwargs.get('percent', 90)/100
    batch_size = kwargs.get('batch_size', 8)

    def valid_feat(item):
        """valid_feat
        Checks if feature is in labels
        :param item: key value pair from read_mat_arkoftmax(logits, dim=1)        """
        return item[0] in train_label_dict
    
    features = []
    labels = []
    # Directly filter out all utterances without labels
    for idx, (k, feat) in enumerate(filter(valid_feat, kaldi_io.read_mat_ark(feature_string))):
        if transform:
            feat = transform(feat)
        features.append(feat)
        labels.append(train_label_dict[k])
    # 90/10 split for training data
    if target_label_name == 'label':
        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_percentage, stratify=labels, random_state=0)
        # Train dataset
        train_dataset = ListDataset(X_train, y_train)
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
        # CV dataset
        cv_dataset = ListDataset(X_test, y_test)
        cv_dataloader = torch.utils.data.DataLoader(
                cv_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    else: #'DAT' or 'DADA'
        outdomain_utt_num = sum(np.array(labels)[:,1]==outdomain_label)
        indomain_utt_num = sum(np.array(labels)[:,1]!=outdomain_label)
        outdomain_features, outdomain_labels = [], []
        indomain_features, indomain_labels = [], []
        for i in range(len(labels)):
            feature = features[i]
            label = labels[i]
            if label[1] == outdomain_label:
                outdomain_features.append(feature)
                outdomain_labels.append(label)
            else:
                indomain_features.append(feature)
                indomain_labels.append(label)
        assert outdomain_utt_num + indomain_utt_num == len(labels), "Outdomain label error!"
        X_train1, X_cv1, y_train1, y_cv1 = train_test_split(outdomain_features, outdomain_labels, train_size=train_percentage, stratify=outdomain_labels, random_state=0)
        # Oversample the train dataset that has fewer samples
        random_oversampler = RandomOverSampler(random_state=0)
        X = X_train1 + indomain_features
        y = y_train1 + indomain_labels
        _, _ = random_oversampler.fit_resample(torch.empty(len(X), 1), np.array(y)[:,1])
        indicies = random_oversampler.sample_indices_
        len1 = len(y_train1)
        print("Outdomain num: {}".format(len1))
        len2 = len(indomain_labels)
        print("Indomain num: {}".format(len2))
        X_train_outdomain, y_train_outdomain = [], []
        X_train_indomain, y_train_indomain = [], []
        for index in indicies:
            feature = X[index]
            label = y[index]
            if label[1] == outdomain_label:
                X_train_outdomain.append(feature)
                y_train_outdomain.append(label)
            else:
                X_train_indomain.append(feature)
                y_train_indomain.append(label)
        assert len(y_train_outdomain)==len(y_train_indomain), "indomain_num != outdomain_num"
        print("Outdomain/indomain num: {}".format(len(y_train_outdomain)))
        
        outdomain_train_dataset = ListDataset(X_train_outdomain, y_train_outdomain)
        outdomain_train_dataloader = torch.utils.data.DataLoader(
                outdomain_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
        indomain_train_dataset = ListDataset(X_train_indomain, y_train_indomain)
        indomain_train_dataloader = torch.utils.data.DataLoader(
                indomain_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
        # CV dataset
        X_cv = X_cv1
        y_cv = y_cv1
        #print("CV Num: {}".format(len(y_cv)))
        #print("CV Labels: {}".format(y_cv))
        cv_dataset = ListDataset(X_cv, y_cv)
        cv_dataloader = torch.utils.data.DataLoader(
                cv_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    if target_label_name == 'label':
        return train_dataloader, cv_dataloader
    else: # 'DAT' or 'DADA'
        return outdomain_train_dataloader, indomain_train_dataloader, cv_dataloader


def criterion_improver(mode):
    """Returns a function to ascertain if criterion did improve

    :mode: can be ether 'loss' or 'acc'
    :returns: function that can be called, function returns true if criterion improved

    """
    assert mode in ('loss', 'acc')
    best_value = np.inf if mode == 'loss' else 0

    def comparator(x, best_x):
        return x < best_x if mode == 'loss' else x > best_x

    def inner(x):
        # rebind parent scope variable
        nonlocal best_value
        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner


# Automatic device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

def main(config='config/train.yaml', **kwargs):
    """Trains a model on the given features and vocab.

    :config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG VALUE
    :returns: None
    """

    config_parameters = parse_config_or_kwargs(config, **kwargs)
    outputdir = os.path.join(
        config_parameters['outputpath'],
        config_parameters['model'],
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))
    try:
        os.makedirs(outputdir)
    except IOError:
        pass
    logger = genlogger(outputdir, 'train.log')
    logger.info("Storing data at: {}".format(outputdir))
    logger.info("<== Passed Arguments ==>")
    # Print arguments into logs
    for line in pformat(config_parameters).split('\n'):
        logger.info(line)

    # seed setting
    seed=config_parameters['seed'] # 1~5
    np.random.seed(seed)
    torch.manual_seed(seed)

    kaldi_string = parsecopyfeats(
        config_parameters['features'], **config_parameters['feature_args'])

    scaler = getattr(pre, config_parameters['scaler'])(**config_parameters['scaler_args'])
    logger.info("<== Estimating Scaler ({}) ==>".format(scaler.__class__.__name__))
    inputdim = -1
    for kid, feat in kaldi_io.read_mat_ark(kaldi_string):
        scaler.partial_fit(feat)
        inputdim = feat.shape[-1]
    assert inputdim > 0, "Reading inputstream failed"
    logger.info(
        "Features: {} Input dimension: {}".format(
            config_parameters['features'],
            inputdim))
    
    logger.info("<== Labels ==>")
    # Can be label, DAT, DADA ... default is 'label'
    target_label_name = config_parameters.get('label_type', 'label')
    if target_label_name == 'label':
        label_df = pd.read_csv(config_parameters['labels'], sep=' ', names=['speaker', 'filename', 'physical', 'system', 'label'])
    else: # 'DAT' or 'DADA'
        label_df = pd.read_csv(config_parameters['labels'], sep=' ', names=['speaker', 'filename', 'physical', 'system', 'label', 'domain'])
    label_encoder = pre.LabelEncoder()
    if target_label_name == 'label':
        label_encoder.fit(label_df[target_label_name].values.astype(str))
        # Labelencoder needs an iterable to work, so just put a list around it and fetch again the 0-th element ( just the encoded string )
        label_df['class_encoded'] = label_df[target_label_name].apply(lambda x: label_encoder.transform([x])[0])
        train_labels = label_df[['filename', 'class_encoded']].set_index('filename').loc[:, 'class_encoded'].to_dict()
    else: # 'DAT' or 'DADA'
        label_encoder_sub = pre.LabelEncoder()
        label_encoder.fit(label_df['label'].values.astype(str))
        label_df['lab_encoded'] = label_df['label'].apply(lambda x: label_encoder.transform([x])[0])
        label_encoder_sub.fit(label_df['domain'].values.astype(str))
        label_df['domain_encoded'] = label_df['domain'].apply(lambda x: label_encoder_sub.transform([x])[0])
        train_labels = label_df[['filename', 'lab_encoded', 'domain_encoded']].set_index('filename').to_dict('index')
        train_labels = {k:np.array(list(v.values())) for k, v in train_labels.items()}
        # outdomain
        outdomain = config_parameters['outdomain']
        outdomain_label = label_encoder_sub.transform([outdomain])[0]
        logger.info("Outdomain: {}, Outdomain label: {}".format(outdomain, outdomain_label))
    
    if target_label_name == 'label':
        train_dataloader, cv_dataloader = create_dataloader_train_cv(kaldi_string, train_labels, transform=scaler.transform, target_label_name=target_label_name, **config_parameters['dataloader_args'])
    else: #'DAT' or 'DADA' 
        outdomain_train_dataloader, indomain_train_dataloader, cv_dataloader = create_dataloader_train_cv(kaldi_string, train_labels, transform=scaler.transform, target_label_name=target_label_name, outdomain_label=outdomain_label, **config_parameters['dataloader_args'])

    if target_label_name == 'label':
        model = getattr(models, config_parameters['model'])(inputdim=inputdim, outputdim=len(label_encoder.classes_), **config_parameters['model_args'])
    else: # 'DAT' or 'DADA'
        model = getattr(models, config_parameters['model'])(inputdim=inputdim, outputdim1=len(label_encoder.classes_), outputdim2=len(label_encoder_sub.classes_), **config_parameters['model_args'])
    logger.info("<== Model ==>")
    for line in pformat(model).split('\n'):
        logger.info(line)
    optimizer = getattr(torch.optim, config_parameters['optimizer'])(model.parameters(), **config_parameters['optimizer_args'])

    scheduler = getattr(torch.optim.lr_scheduler, config_parameters['scheduler'])(optimizer, **config_parameters['scheduler_args'])
    criterion = getattr(loss, config_parameters['loss'])(**config_parameters['loss_args'])
    trainedmodelpath = os.path.join(outputdir, 'model.th')

    model = model.to(device)
    criterion_improved = criterion_improver(config_parameters['improvecriterion'])
    header = [
        'Epoch',
        'Lr',
        'Loss(T)',
        'Loss(CV)',
        "Acc(T)",
        "Acc(CV)",
    ]
    for line in tp.header(header, style='grid').split('\n'):
        logger.info(line)

    for epoch in range(1, config_parameters['epochs']+1):
        if target_label_name == 'label':
            train_loss, train_acc = runepoch(train_dataloader, None, model, criterion, target_label_name, optimizer, dotrain=True, epoch=epoch)
        else: # 'DAT' or 'DADA'
            train_loss, train_acc = runepoch(outdomain_train_dataloader, indomain_train_dataloader, model, criterion, target_label_name, optimizer, dotrain=True, epoch=epoch)
        cv_loss, cv_acc = runepoch(cv_dataloader, None, model, criterion, target_label_name, dotrain=False, epoch=epoch)
        logger.info(
            tp.row(
                (epoch,) + (optimizer.param_groups[0]['lr'],) +
                (str(train_loss), str(cv_loss), str(train_acc), str(cv_acc)),
                style='grid'))
        epoch_meanloss = cv_loss[0] if type(cv_loss)==tuple else cv_loss
        if epoch % config_parameters['saveinterval'] == 0:
            torch.save({'model': model,
                        'scaler': scaler,
                        'encoder': label_encoder,
                        'config': config_parameters},
                        os.path.join(outputdir, 'model_{}.th'.format(epoch)))
        # ReduceOnPlateau needs a value to work
        schedarg = epoch_meanloss if scheduler.__class__.__name__ == 'ReduceLROnPlateau' else None
        scheduler.step(schedarg)
        if criterion_improved(epoch_meanloss):
            torch.save({'model': model,
                        'scaler': scaler,
                        'encoder': label_encoder,
                        'config': config_parameters},
                        trainedmodelpath)
        if optimizer.param_groups[0]['lr'] < 1e-7:
            break
    logger.info(tp.bottom(len(header), style='grid'))
    logger.info("Results are in: {}".format(outputdir))


def run_evaluation(scores_file: str, evaluation_res_file: str):
    # Directly run the evaluation
    SCRIPT_FILE = "./scripts/tDCF_python_v1/evaluate_tDCF_asvspoof19.py"
    ASV_MODEL_RES = f"./scripts/tDCF_python_v1/scores/asv_dev.txt"
    os.system("python3 {} {} {} | tee {}".format(
        SCRIPT_FILE, scores_file, ASV_MODEL_RES, evaluation_res_file))
    print(
        f"Evaluation results are at {evaluation_res_file}\n Prediction scores are at {scores_file}")


def forward_model(model_path: str, features: str,
                  label_type: str = 'label',
                  # The output scores file ( stored in model_path )
                  class_name: str = 'bonafide',
                  log_softmax: bool = False,
                  softmax: bool = False,
                  with_cl_diff: bool = False,
                  **kwargs):
    model_dump = torch.load(model_path, lambda storage, loc: storage)
    config_parameters = model_dump['config']
    model = model_dump['model']
    scaler = model_dump['scaler']
    encoder = model_dump['encoder']
    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])
    # Get the class_idx belonging to the class we represent for true scores
    try:
        class_idx = encoder.transform([class_name])[0]
    except ValueError as e:
        # Error happens when training more than one class and fail to adjust class_name
        # However, we propose the other possible classes stored in the encoder
        print(e)
        print("Available classes are {}".format(encoder.classes_))
        return
    model.to(device).eval()
    outputs = {}
    with torch.no_grad():
        for key, feat in kaldi_io.read_mat_ark(kaldi_string):
            feat = scaler.transform(feat)
            # Add singleton batch dim
            feat = torch.from_numpy(feat).unsqueeze(0).to(device)
            # Forward through model
            output = model(feat)

            if label_type == 'DAT' or label_type == 'DADA':
                out1, out2 = output
                if log_softmax:
                    out1, out2 = F.log_softmax(out1, dim=-1), F.log_softmax(out2, dim=-1)
                elif softmax:
                    out1, out2 = F.softmax(out1, dim=-1), F.softmax(out2, dim=-1)
                out1, out2 = out1.squeeze(0).cpu().detach().numpy(), out2.squeeze(0).cpu().detach().numpy()
                if with_cl_diff:
                    outputs[key] = out1[class_idx] - np.mean(out1[~class_idx])
                else:
                    outputs[key] = out1[class_idx]
            else:
                if log_softmax:
                    output = F.log_softmax(output, dim=-1)
                if softmax:
                    output = F.softmax(output, dim=-1)
                output = output.squeeze(0).cpu().detach().numpy()
                if with_cl_diff:
                    outputs[key] = output[class_idx] - np.mean(output[~class_idx])
                else:
                    outputs[key] = output[class_idx]
    return outputs


def test_model(model_path: str, features: str,
               label_type: str = 'label',
               # The output scores file ( stored in model_path )
               output: str = 'predictions.txt',
               class_name: str = 'bonafide',  # The target class representing Genuine
               dataset: str = "ASV17",  # Can be either ASV17 or BTAS16-PA
               eval_out: str = "evaluation.txt",  # The output of tDCF evaluation script
               log_softmax: bool = False,  # Use or not to use log_softmax at the final layer
               softmax: bool = False, # Passed to forward
               with_cl_diff: bool = False, # Passed to forward
               **kwargs):
    
    assert dataset in ("ASV17", "BTAS16-PA"), "dataset needs to be either ASV17 or BTAS16-PA"
    if dataset in ("BTAS16-PA_eval"):
        test_label_file = "./data/BTAS16-PA/labels/eval.txt"
    else: # dataset in ("ASV17_eval"):
        test_label_file = "./data/ASV17/labels/eval.txt"

    test_df = pd.read_csv(test_label_file, names=['speaker', 'filename', 'physical', 'system', 'label'], sep=' ')
    arkkeys_to_outputprob = forward_model(model_path=model_path, features=features,  label_type=label_type, log_softmax=log_softmax, class_name=class_name, with_cl_diff=with_cl_diff, softmax= softmax)
    scores_file = os.path.join(os.path.dirname(model_path), output)
    with open(scores_file, 'w') as wp:
        for key, keyoutput in arkkeys_to_outputprob.items():
            if not test_df.filename.isin([key]).any():
                print("Key {} not found in labels file {}".format(key, test_label_file))
                continue
            sys, label = test_df[test_df.filename == key][['system', 'label']].values[0]
            key = Path(key).resolve().stem
            wp.write("{} {} {} {}\n".format(key, sys, label, keyoutput))
    evaluation_res_file = os.path.join(os.path.dirname(model_path), eval_out)
    # Runs just the tDCF evaluations cript and puts results into evaluation_res_file
    run_evaluation(scores_file, evaluation_res_file)


def extract_embeddings_from_model(model_path: str, features: str, output_arkfile: str, **kwargs):
    model_dump = torch.load(model_path, lambda storage, loc: storage)
    config_parameters = model_dump['config']
    model = model_dump['model']
    scaler = model_dump['scaler']
    encoder = model_dump['encoder']
    feature_string = config_parameters['features'] if not features else features
    kaldi_string = parsecopyfeats(
        feature_string, **config_parameters['feature_args'])

    arkfile = os.path.join(os.path.dirname(model_path), output_arkfile)
    model.to(device).eval()
    with torch.no_grad():
        with open(arkfile, "wb") as fp:
            for key, feat in kaldi_io.read_mat_ark(kaldi_string):
                feat = scaler.transform(feat)
                # Add singleton batch dim
                feat = torch.from_numpy(feat).unsqueeze(0).to(device)
                # Forward through model
                output = model.extract(feat)
                embedding = output.squeeze(0).cpu().numpy()
                kaldi_io.write_vec_flt(fp, embedding, key=key) 


if __name__ == '__main__':
    fire.Fire({
        'train': main,
        'score': test_model,
        'forward': forward_model,
        'extract': extract_embeddings_from_model,
    })
