import datafree
from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import collections
import random
import os
import copy

from torchvision import utils
from torch.autograd import Variable
from torch import autograd
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv, cross_entropy, max_margin_loss
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
from tqdm import tqdm
from InvertedData import InvertedData

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

class DFATWGanSynthesizer(BaseSynthesis):
    def __init__(self, model, nz, num_classes, img_size, d_dataset,
                 feature_layers=None, bank_size=40960, n_neg=4096, head_dim=128, init_dataset=None,
                 iterations=100, lr_g=0.001,lr_d=0.001, lr_z=0.001, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0.0, bn=1, oh=1, cr=0.8, cr_T=0.1, alpha = 0.1, 
                 save_dir='run/cmi', transform=None,
                 autocast=None, use_fp16=False, pretrained=False,
                 normalizer=None, device='cpu', distributed=False):
        super(DFATWGanSynthesizer, self).__init__(model,model)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lr_z = lr_z
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.n_neg = n_neg
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.alpha = alpha
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.bank_size = bank_size
        self.init_dataset = init_dataset

        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.d_dataset = InvertedData(d_dataset, transforms=self.transform)
        self.d_dataloader = torch.utils.data.DataLoader(self.d_dataset,
                                                        batch_size=synthesis_batch_size, shuffle=True,
                                                        num_workers=4, pin_memory=True)
        self.data_iter = None
        self.cr = cr
        self.cr_T = cr_T
        self.cmi_hooks = []
        self.generator_iters = 8000
        self.critic_iter = 5
        self.lambda_term = 10
        if feature_layers is not None:
            for layer in feature_layers:
                self.cmi_hooks.append( InstanceMeanHook(layer) )
        else:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.cmi_hooks.append( InstanceMeanHook(m) )

        with torch.no_grad():
            model.eval()
            fake_inputs = torch.randn(size=(1, *img_size), device=device)
            _ = model(fake_inputs)
            cmi_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
            print("CMI dims: %d"%(cmi_feature.shape[1]))
            del fake_inputs
        self.generator = datafree.models.generator.WGAN_GP_Generator(nz=nz,nc=3).to(device).train()
        self.discriminator = datafree.models.generator.WGAN_GP_Discriminator(nc=3).to(device).train()
        if pretrained:
            self.generator.load_state_dict(torch.load("/home/mazhongming/DFAT/DFAT/checkpoints/wgan/generator.pkl"))
            self.discriminator.load_state_dict(torch.load("/home/mazhongming/DFAT/DFAT/checkpoints/wgan/discriminator.pkl"))
            print("Loaded checkpoint!")
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        # local and global bank
        self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1]) # local + global
        
        self.head = MLPHead(cmi_feature.shape[1], head_dim).to(device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_g)

        self.device = device
        self.hooks = []
        self.synthesize_iter = 0
        for m in model.modules():
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

    def get_infinite_batches(self, data_loader):
        while True:
            for _, (images,_) in enumerate(data_loader):
                yield images

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.synthesis_batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.synthesis_batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        interpolated = interpolated.to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def save_model(self):
        torch.save(self.generator.state_dict(), 'checkpoints/wgan/generator.pkl')
        torch.save(self.discriminator.state_dict(), 'checkpoints/wgan/discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def get_torch_variable(self, arg):
        return Variable(arg).to(self.device)

    def train(self):
        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(self.d_dataloader)
        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = one * -1

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.discriminator.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.discriminator.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.synthesis_batch_size):
                    continue

                images = self.get_torch_variable(images)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.discriminator(self.normalizer(images))
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.synthesis_batch_size, self.nz, 1, 1))

                fake_images = self.generator(z)
                d_loss_fake = self.discriminator(self.normalizer(fake_images))
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(self.normalizer(images.data), self.normalizer(fake_images.data))
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real} Wasserstein_D: {Wasserstein_D:.2f}')

            # Generator update
            for p in self.discriminator.parameters():
                p.requires_grad = False  # to avoid computation

            self.generator.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.synthesis_batch_size, self.nz, 1, 1))
            fake_images = self.generator(z)
            g_loss = self.discriminator(self.normalizer(fake_images))
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % 100 == 0:
                self.save_model()
                os.makedirs('training_result_images/',exist_ok=True)

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(self.synthesis_batch_size, self.nz, 1, 1))
                samples = self.generator(z)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                print("Generator iter: {}".format(g_iter))

        self.save_model()


    def synthesize(self, targets=None):
        # self.student.eval()
        self.synthesize_iter += 1
        self.teacher.eval()
        best_cost = 1e6
        
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz,1,1), device=self.device).requires_grad_() 
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)
        generator = copy.deepcopy(self.generator)
        for p in self.discriminator.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam([{'params': generator.parameters()}, {'params': [z]}], self.lr_z, betas=[0.5, 0.999])
        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = one * -1
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.1*self.lr)
        for it in tqdm(range(self.iterations)):
            #############################################
            # Discriminator Loss
            #############################################
            inputs = generator(z)
            d_loss_fake = self.discriminator(self.normalizer(inputs))
            d_loss_fake = d_loss_fake.mean()

            global_view, local_view = self.aug(inputs) # crop and normalize
            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(global_view)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = cross_entropy( t_out, targets )
            loss_mm = max_margin_loss( t_out, targets )
            if self.adv>0:
                with torch.no_grad():
                    random_noise = torch.FloatTensor(*inputs.shape).uniform_(-8/255, 8/255).to(self.device)
                    adv_inputs = (random_noise + inputs).clamp(0,1.0)
                    adv_global_view, _ = self.aug(adv_inputs)
                adv_out = self.teacher(adv_global_view)
                loss_adv = cross_entropy(adv_out, t_out.max(1)[1], reduction='mean')
            else:
                loss_adv = loss_oh.new_zeros(1)

            loss_inv = self.bn * loss_bn + self.oh * loss_oh +  self.adv * loss_adv + self.alpha * loss_mm * max(1-it/self.iterations,0.1)
 
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
            loss_cr = cross_entropy( cr_logits, cr_labels, reduction='none')  #(N + N')
            if self.mem_bank.n_updates>0:
                loss_cr = loss_cr[:self.synthesis_batch_size].mean() + loss_cr[self.synthesis_batch_size:].mean()
            else:
                loss_cr = loss_cr.mean()
            
            if d_loss_fake > 3000:
                loss = self.cr * loss_cr + loss_inv 
            else:
                loss = self.cr * loss_cr + loss_inv - d_loss_fake*0.01
            # print(f"Loss_oh:{loss_oh}")
            # print(f"Loss_inv:{loss_inv}")
            # print(f"Loss_cr:{loss_cr}")
            # print(f"Loss_d:{d_loss_fake}")
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
                    best_features = torch.cat([local_feature.data, global_feature.data], dim=1).data
            optimizer.zero_grad()
            self.optimizer_head.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_head.step()

        # self.student.train()
        # save best inputs and reset data iter
        samples = best_inputs.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'training_result_images/img_syn_iter_{}.png'.format(str(self.synthesize_iter).zfill(3)))
        self.data_pool.add( best_inputs ,target=targets)
        self.mem_bank.add( best_features )

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