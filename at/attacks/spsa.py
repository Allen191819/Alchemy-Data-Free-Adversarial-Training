import numpy as np

import torch
import advertorch
from autoattack.fab_pt import FABAttack_PT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinfSPSA():
    def __init__(self, predict, loss_fn='ce', eps=0.3, nb_iter=40, ord=np.inf, seed=1,device=device):
        self.device= device
        assert ord in [np.inf], 'Only ord=inf is supported!'
        
        self.spsa = advertorch.attacks.LinfSPSAAttack(predict=predict,eps=eps,nb_iter=nb_iter,max_batch_size=32,lr=0.01,targeted=False,nb_sample=128,loss_fn=None)


    def perturb(self, x, y):
        x, y = x.to(self.device),y.to(self.device)
        x_adv = self.spsa.perturb(x, y)
        r_adv = x_adv - x
        return x_adv, r_adv