import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''
files to devise different loss strategy !
'''

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()      # multiply flags and labels
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.5, gamma = 2, logits = False, reduce = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class weighted_entropy(nn.Module):
    '''
    pred  : N, C,
    label : N, -1
    '''
    def __init__(self, need_soft_max = True):
        super(weighted_entropy, self).__init__()
        self.need_soft_max = need_soft_max
        pass

    def forward(self, pred, label):
        if self.need_soft_max is True:
            preds = F.softmax(pred, dim=1)
        else:
            preds = pred
        epusi  = 1e-10
        counts = torch.rand(size=(2,))
        counts[0] = label[torch.where(label == 0)].size(0)
        counts[1] = label[torch.where(label == 1)].size(0)
        N = label.size()[0]
        weights = counts[1]
        weights_avg = 1 - weights / N
        loss = weights_avg * torch.log(preds[:,1] + epusi) + (1 - weights_avg) * torch.log(1 - preds[:,1] + epusi)
        loss = - torch.mean(loss)
        return loss

def loss_ce(preds, masks, criterion, selected_mode = 1):

    '''
    :param preds:       N  H  W  C
    :param masks:       N  1  H  W
    :param criterion:   This is used to calculate nn.cross-entropy() or nn.BCE-loss(), both is OK !
    :return:            criterion (N*H*W, C  and  N,-1)
    '''
    if selected_mode == 1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs = preds.permute((0, 2, 3, 1))         # N H W C
    outs = outs.reshape((-1, outs.size()[3]))  # N*H*W, C
    if selected_mode == 1:
        outs = outs.reshape((-1,))
    masks = masks.reshape((-1,))               # N,1,H,W ===> N,-1
    if selected_mode == 2:
        masks = torch.tensor(masks, dtype=torch.long)
    return criterion(outs, masks)



def loss_ce_ds(preds, masks, criterion, selected_mode = 2):
    # this is used to calculate cross-entropy with many categories !
    if selected_mode ==1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs0 = preds[0].permute((0, 2, 3, 1))  # N H W C
    outs0 = outs0.reshape((-1, outs0.size()[3]))  # N*H*W, C
    outs1 = preds[1].permute((0, 2, 3, 1))  # N H W C
    outs1 = outs1.reshape((-1, outs1.size()[3]))  # N*H*W, C
    outs2 = preds[2].permute((0, 2, 3, 1))  # N H W C
    outs2 = outs2.reshape((-1, outs2.size()[3]))  # N*H*W, C
    outs3 = preds[3].permute((0, 2, 3, 1))  # N H W C
    outs3 = outs3.reshape((-1, outs3.size()[3]))  # N*H*W, C
    masks = masks.reshape((-1,))  # N,1,H,W ===> N,-1
    masks = torch.tensor(masks, dtype=torch.long)
    loss = 0.25 * criterion(outs0, masks) + 0.5 * criterion(outs1, masks) + \
           0.75 * criterion(outs2, masks) + 1.0 * criterion(outs3, masks)
    return loss

if __name__ == '__main__':
    labels = torch.tensor([0, 1, 1, 0, 1, 1])
    pred = torch.tensor([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.4, 0.6], [0.3, 0.7], [0.3, 0.7]])
    pred2 = torch.tensor([0.3, 0.7, 0.6, 0.2, 0.5, 0.9])

    print(weighted_entropy(need_soft_max = False)(pred,labels))
    print(DiceLoss()(pred2, labels))
    print(FocalLoss()(pred2, torch.tensor(labels, dtype=torch.float)))
    print(nn.CrossEntropyLoss()(pred, labels))

    pass