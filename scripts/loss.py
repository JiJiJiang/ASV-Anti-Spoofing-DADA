import torch
import torch.nn as nn
import torch.nn.functional as F


BCELoss = nn.BCELoss
BCEWLLoss = nn.BCEWithLogitsLoss
CrossEntropyLoss = nn.CrossEntropyLoss
