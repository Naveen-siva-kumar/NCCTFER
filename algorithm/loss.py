'''
Aum Sri Sai Ram

Naveen
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F

# import math
eps = 1e-8




class Sai_weighted_CCE(nn.Module):
    """
    Implementing Noise Robust CE Loss 
    
    """

    def __init__(self,  num_class=7,device=None, reduction="mean"):
        super(Sai_weighted_CCE, self).__init__()
        
        self.reduction = reduction
        self.num_class= num_class
        self.device = device

    def forward(self, prediction, target_label,complemetary_labels=None,n_class=False, one_hot=True,eps=0.35):
        if(n_class):
            one_hot = False
            
            y_true = (1.0-F.one_hot(target_label,self.num_class).float()).to(self.device)
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).float().to(self.device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)
        
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)
        
        avg_post = torch.mean(y_pred, dim=0)
        avg_post = avg_post.reshape(-1, 1)
        std_post = torch.std(y_pred, dim=0)
        std_post = std_post.reshape(-1, 1)
        
        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        #std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        pred_prun = torch.where((pred_tmp >= avg_post_ref ), pred_tmp, torch.zeros_like(pred_tmp)) #confident
        confident_idx = torch.where(pred_prun != 0.)[0]
        noisy_idx = torch.where(pred_prun == 0.)[0]
        if(n_class):
            labels = complemetary_labels
        else:
            
        
            N, C = prediction.shape
        
        
            smooth_labels = torch.full(size=(N, C), fill_value=eps/(C - 1)).to(self.device)
            smooth_labels.scatter_(dim=1, index=torch.unsqueeze(target_label, dim=1), value=1-eps)
            labels =  smooth_labels
        if len(confident_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, confident_idx), dim=1)
            #prun_targets = labels[confident_idx]
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, confident_idx), 
                            prun_targets, reduction="sum")
        else:
            labels = target_label
            weighted_loss = F.cross_entropy(prediction, labels)

        return weighted_loss,noisy_idx

        
def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')        
        
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

    
    
class Sai_weighted_CCE_(nn.Module):
    """
    Implementing Noise Robust CE Loss 
    
    """

    def __init__(self,  num_class=7,device=None, reduction="mean"):
        super(Sai_weighted_CCE_, self).__init__()
        
        self.reduction = reduction
        self.num_class= num_class
        self.device = device

    def forward(self, prediction, target_label,complemetary_labels=None,n_class=False, one_hot=True,eps=0.35):
        if(n_class):
            one_hot = False
            y_true = (1.0-F.one_hot(target_label,self.num_class).float()).to(self.device)
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).float().to(self.device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)
        
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)
        
        avg_post = torch.mean(y_pred, dim=0)
        avg_post = avg_post.reshape(-1, 1)
        std_post = torch.std(y_pred, dim=0)
        std_post = std_post.reshape(-1, 1)
        
        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        
        pred_prun = torch.where((pred_tmp >= avg_post_ref ), pred_tmp, torch.zeros_like(pred_tmp)) #confident
        confident_idx = torch.where(pred_prun != 0.)[0]
        noisy_idx = torch.where(pred_prun == 0.)[0]
        if(n_class):
            labels = complemetary_labels
        else:
            
        
            N, C = prediction.shape
        
        
            smooth_labels = torch.full(size=(N, C), fill_value=eps/(C - 1)).to(self.device)
            smooth_labels.scatter_(dim=1, index=torch.unsqueeze(target_label, dim=1), value=1-eps)
            labels =  smooth_labels
        if len(confident_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, confident_idx), dim=1)
           
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, confident_idx), 
                            prun_targets, reduction="sum")
        else:
            labels = target_label
            weighted_loss = F.cross_entropy(prediction, labels)

        return weighted_loss,noisy_idx,confident_idx

