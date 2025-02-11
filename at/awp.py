import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import OrderedDict

EPS = 1E-20

def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma, epochs,normalizer):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.epochs = epochs
        self.normalizer = normalizer

    def calc_awp(self, inputs_adv, inputs_clean, targets, epoch ,beta, datafree=True):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        logits = self.proxy(self.normalizer(inputs_clean))
        logits_adv = self.proxy(self.normalizer(inputs_adv))

        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits, dim=1),
                               reduction='batchmean')
        # loss_robust = F.cross_entropy(logits_adv,targets.max(1)[1])

        # calculate natural loss and backprop
        if datafree:
            loss_natural = F.kl_div(F.log_softmax(logits,dim=1),
                                    F.softmax(targets,dim=1),
                                    reduction='batchmean')
        else:
            loss_natural = F.cross_entropy(logits,targets)
        # loss_natural = loss_kl*(1 - epoch/self.epochs) + loss_ce*(epoch/self.epochs)
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

