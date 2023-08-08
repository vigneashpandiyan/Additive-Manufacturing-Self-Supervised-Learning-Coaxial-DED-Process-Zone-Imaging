import torch
import torch.nn as nn
import torch.nn.functional as F


loss_fn = nn.CrossEntropyLoss(reduction='sum')

