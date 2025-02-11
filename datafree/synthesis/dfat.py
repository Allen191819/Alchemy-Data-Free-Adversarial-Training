import datafree
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import copy
import collections

from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Generator
from torch import optim
from tqdm import tqdm
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv, max_margin_loss, cross_entropy
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation

class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)

class MultiTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str( self.transform )


class ContrastLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, return_logits=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if return_logits:
            return loss, anchor_dot_contrast
        return loss

class FidBank(object):
    def __init__(self,num_classes,feature,max_images,device):
        self.num_classes = num_classes
        self.max_images = max_images
        self.images_groups = [None for _ in range(num_classes)]
        self.fid = FrechetInceptionDistance(feature=feature).to(device)

    def add_images(self,label,images):
        if self.images_groups[label] is None:
            self.images_groups[label] = images
        else:
            self.images_groups[label] = torch.cat([self.images_groups[label],images])
            if self.images_groups[label].shape[0] > self.max_images:
                self.images_groups[label] = self.images_groups[label][-self.max_images:]

    def get_fid(self,label,images):
        self.add_images(label,images)
        if None in self.images_groups:
            return 0
        else:
            fid_score = []
            targets = random.sample(list(filter(lambda x:x!=label,[i for i in range(10)])),self.num_classes//3)
            for l in targets:
                self.fid.reset()
                self.fid.update((self.images_groups[label].clamp(0,1.0)*255).to(torch.uint8),real=False)
                self.fid.update((self.images_groups[l].clamp(0,1.0)*255).to(torch.uint8),real=True)
                fid_score.append(self.fid.compute())
            return - torch.Tensor(fid_score).mean()


class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        feat = feat.to(self.device)
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(self.dim_feat, c, self.max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            #return self.data[:self._ptr]
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class DFATSynthesizer(BaseSynthesis):
    def __init__(self, model, nz, num_classes, img_size, 
                 feature_layers=None, bank_size=40960, n_neg=4096, head_dim=128, init_dataset=None,
                 iterations=100, iterations_g=1000,lr_g=0.001, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128, lr_z=0.001,
                 adv=0.1, bn=0.2, oh=1, cr=0.8, cr_T=0.1,alpha=1,mm=0.1,train_loss = "ce",
                 save_dir='run/cmi', transform=None,
                 autocast=None, use_fp16=False, pre_trained=False,
                 normalizer=None, device='cpu', distributed=False,debug=False):
        super(DFATSynthesizer, self).__init__(model, model)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.iterations_g = iterations_g
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.n_neg = n_neg
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.mm = mm
        self.alpha = alpha
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.bank_size = bank_size
        self.init_dataset = init_dataset
        self.fid_bank = FidBank(num_classes,64,256,device)
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        if train_loss == "mm":
            self.criterion = max_margin_loss
        elif train_loss == "ce":
            self.criterion = cross_entropy
        else:
            raise Exception(Exception("Invalid Loss Option!", train_loss))
        self.cr = cr
        self.cr_T = cr_T
        self.cmi_hooks = []
        if feature_layers is not None:
            for layer in feature_layers:
                self.cmi_hooks.append( InstanceMeanHook(layer) )
        else:
            for m in self.teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.cmi_hooks.append( InstanceMeanHook(m) )

        with torch.no_grad():
            self.teacher.eval()
            fake_inputs = torch.randn(size=(1, *img_size), device=device)
            _ = self.teacher(fake_inputs)
            cmi_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
            print("CMI dims: %d"%(cmi_feature.shape[1]))
            del fake_inputs
        
        self.feature_size = cmi_feature.shape[1] 
        self.generators = []
        for i in range(num_classes):
            self.generators.append(datafree.models.generator.Generator(nz=nz, ngf=64, img_size=img_size[1], nc=3).to(device).train())
            if pre_trained:
                checkpoint = torch.load(f"./checkpoints/generators/G_class_{i:02d}.pth", map_location=device)
                self.generators[i].load_state_dict(checkpoint['state_dict'])
        if pre_trained:
            print(f"Pretrained model loaded!")
        # local and global bank
        self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1]) # local + global
        
        self.head = MLPHead(cmi_feature.shape[1], head_dim).to(device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_g)

        self.device = device
        self.debug = debug
        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

        self.aug = MultiTransform([
            # global view
            transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
        ])

        #self.contrast_loss = ContrastLoss(temperature=self.cr_T, contrast_mode='one')

    def train_generator(self,targets=None):
        self.teacher.eval()
        if targets is None:
            targets = [i for i in range(self.num_classes)]
        print(f"Training generator ==> {targets}")
        for target in targets:
            self.generators[target].train()
            reset_model(self.generators[target])
            target_G = torch.LongTensor([target,]*self.synthesis_batch_size).to(self.device)
            # optim_G = torch.optim.SGD(self.generators[target].parameters(), self.lr_g, weight_decay=1e-4, momentum=0.9)
            optim_G = torch.optim.Adam(self.generators[target].parameters(), self.lr_g, betas=[0.5, 0.999])
            # scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G, T_max=self.iterations_g, eta_min=0.01*self.lr_g)
            for it in tqdm(range(self.iterations_g),desc=f"Training generator ==> {target}"):
                z = torch.randn(size=(self.synthesis_batch_size,self.nz),device=self.device)

                inputs = self.generators[target](z)
                global_view, local_view = self.aug(inputs) # crop and normalize
                t_out = self.teacher(global_view)
                loss_bn = sum([h.r_feature for h in self.hooks])
                loss_oh = self.criterion(t_out,target_G)

                random_noise = torch.FloatTensor(*inputs.shape).uniform_(-16/255, 16/255).to(self.device)
                adv_inputs = (random_noise + inputs).clamp(0,1.0)
                adv_global_view, _ = self.aug(adv_inputs)
                adv_out = self.teacher(adv_global_view)
                loss_no = cross_entropy(adv_out, t_out.max(1)[1], reduction='mean')

                loss = loss_bn*self.bn*0.3 + loss_oh*self.oh + loss_no*self.adv

                optim_G.zero_grad()
                loss.backward()
                optim_G.step()
                # scheduler_G.step()
                _,pred = t_out.max(1)
                if self.debug and it%500==0:
                    self.data_pool.add( inputs )
                    print(f"({it}/{self.iterations_g}) :==: Correct ==> ({(pred == target_G).sum().item()}/{t_out.shape[0]}) Loss:{loss:.4}")
            print(f"Finished training generator ==> {target}")
            os.makedirs("./checkpoints/generators",exist_ok=True)
            torch.save({'state_dict': self.generators[target].state_dict()},f"./checkpoints/generators/G_class_{target:02d}.pth")
        
        print(f"Finished training generator ==> {targets}")

    def synthesize(self):
        self.teacher.eval()
        best_cost = [1e6 for _ in range(self.num_classes)]
        best_inputs = [None for _ in range(self.num_classes)]
        best_features = [None for _ in range(self.num_classes)]
        targets = []
        generators = []
        for label in range(self.num_classes):
            targets.append(torch.LongTensor([label,]*self.synthesis_batch_size).to(self.device))
            generators.append(copy.deepcopy(self.generators[label]))
            # reset_model(generators[label])
        for label in range(self.num_classes):
            z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
            optimizer = torch.optim.Adam([{'params': generators[label].parameters()},{"params":[z]}], self.lr_z, betas=[0.5, 0.999])
            generators[label].train()
            for it in tqdm(range(self.iterations),desc=f"Synthesizing class {label}......"):
                inputs = generators[label](z)
                global_view, local_view = self.aug(inputs)
                #############################################
                # Inversion Loss
                #############################################
                t_out = self.teacher(global_view)
                loss_bn = sum([h.r_feature for h in self.hooks])
                loss_oh = cross_entropy( t_out, targets[label] )
                loss_mm = max_margin_loss( t_out, targets[label] )

                if self.adv>0:
                    random_noise = torch.FloatTensor(*inputs.shape).uniform_(-16/255, 16/255).to(self.device)
                    adv_inputs = (random_noise + inputs).clamp(0,1.0)
                    adv_global_view, _ = self.aug(adv_inputs)
                    adv_out = self.teacher(adv_global_view)
                    loss_adv = cross_entropy(adv_out, t_out.max(1)[1], reduction='mean')
                else:
                    loss_adv = loss_oh.new_zeros(1)

                loss_inv = self.bn * loss_bn + self.oh * loss_oh +  self.adv * loss_adv + self.mm * loss_mm
    
                #############################################
                # Contrastive Loss
                #############################################
                global_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1) 
                _ = self.teacher(local_view)
                local_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1) 
                cached_feature, _ = self.mem_bank.get_data(self.n_neg)
                cached_local_feature, cached_global_feature = torch.chunk(cached_feature.to(self.device), chunks=2, dim=1)

                proj_feature = self.head( torch.cat([local_feature, cached_local_feature, global_feature, cached_global_feature], dim=0) )
                proj_local_feature, proj_global_feature = torch.chunk(proj_feature, chunks=2, dim=0)
                
                # https://github.com/HobbitLong/SupContrast/blob/master/losses.py
                #cr_feature = torch.cat( [proj_local_feature.unsqueeze(1), proj_global_feature.unsqueeze(1).detach()], dim=1 )
                #loss_cr = self.contrast_loss(cr_feature)
                
                # Note that the cross entropy loss will be divided by the total batch size (current batch + cached batch)
                # we split the cross entropy loss to avoid too small gradients w.r.t the generator
                #if self.mem_bank.n_updates>0:
                            # 1. gradient from current batch              +  2. gradient from cached data
                #    loss_cr = loss_cr[:, :self.synthesis_batch_size].mean() + loss_cr[:, self.synthesis_batch_size:].mean()
                #else: # 1. gradients only come from current batch      
                #    loss_cr = loss_cr.mean()

                # A naive implementation of contrastive loss
                cr_logits = torch.mm(proj_local_feature, proj_global_feature.detach().T) / self.cr_T # (N + N') x (N + N')
                cr_labels = torch.arange(start=0, end=len(cr_logits), device=self.device)
                loss_cr = F.cross_entropy( cr_logits, cr_labels, reduction='none')  #(N + N')
                if self.mem_bank.n_updates>0:
                    loss_cr = loss_cr[:self.synthesis_batch_size].mean() + loss_cr[self.synthesis_batch_size:].mean()
                else:
                    loss_cr = loss_cr.mean()

                if self.alpha>0:
                    loss_fid = self.fid_bank.get_fid(label=label,images=inputs)
                else:
                    loss_fid = loss_oh.new_zeros(1)
                loss = self.cr * loss_cr + loss_inv + self.alpha*loss_fid

                # print(f"Loss oh {loss_oh}")
                # print(f"Loss cr {loss_cr}")
                # print(f"Loss adv {loss_adv}")
                # print(f"Loss bn {loss_bn}")
                # print(f"Loss fid {loss_fid}")
                # print(f"Loss inv {loss_inv}")
                # print(f"Loss {loss}")
                
                with torch.no_grad():
                    if best_cost[label] > loss.item() or best_inputs[label] is None:
                        best_cost[label] = loss.item()
                        best_inputs[label] = inputs.data
                        best_features[label] = torch.cat([local_feature.data, global_feature.data], dim=1).data
                optimizer.zero_grad()
                self.optimizer_head.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_head.step()
            self.data_pool.add( best_inputs[label] ,target=label)
            self.mem_bank.add( best_features[label] )

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}
        
    def sample(self):
        return self.data_iter.next()