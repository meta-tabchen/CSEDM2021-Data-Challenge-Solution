#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################

import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from enum import Enum

def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


# https://github.com/yiyixuxu/polyloss-pytorch
class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = False,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: the shape should be BNH[WD] (one-hot format) or B1H[WD], where N is the number of classes.
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
          # squeeze out the channel dimension of size 1 to calculate ce loss
          self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
          # convert into one-hot format to calculate ce loss
          target = to_one_hot(target, num_classes=n_pred_ch)
        else:
          # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
          self.ce_loss = self.cross_entropy(input,torch.argmax(target, dim=1))

        if self.softmax:
          if n_pred_ch == 1:
            warnings.warn("single channel prediction, `softmax=True` ignored.")
          else:
            input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1) # BH[WD]
        poly_loss = self.ce_loss +  self.epsilon * (1 - pt)


        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction =='none':
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            # BH[WD] -> BH1[WD]
            polyl = poly_loss.unsqueeze(1)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return(polyl)
    
def poly_loss(y_pred,y_true,reduction='mean',softmax=True,epsilon=1.0):
    print(y_pred.shape)
    print(y_true.shape)
    loss_fun = PolyLoss(reduction=reduction,softmax=softmax,epsilon=epsilon)
    loss_poly = loss_fun(y_pred,y_true)
    return loss_poly