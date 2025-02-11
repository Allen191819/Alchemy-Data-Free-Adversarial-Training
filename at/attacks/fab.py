import numpy as np

import torch
from autoattack.fab_pt import FABAttack_PT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FAB():
    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, ord=np.inf, seed=1,device=device):
        self.device= device
        assert loss_fn in ['ce', 'dlr'], 'Only loss_fn=ce or loss_fn=dlr are supported!'
        assert ord in [2, np.inf], 'Only ord=inf or ord=2 are supported!'
        
        norm = 'Linf' if ord == np.inf else 'L2'
        self.fab = FABAttack_PT(predict=predict,norm=norm,n_restarts=n_restarts,n_iter=nb_iter,eps=eps,seed=seed,device=device)

    def perturb(self, x, y):
        x_adv = self.fab.perturb(x, y)
        r_adv = x_adv - x
        return x_adv, r_adv