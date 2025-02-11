import datafree
from typing import Generator
import torch
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
import lmdb
import gc
import pyarrow as pa
from datafree.datasets.lmdbDataset0809  import LMDBFeatureLabelImageDataset0809
from datafree.datasets.lmdbDataset1024  import LMDBFeatureLabelImageDataset1024
import pickle5,pickle
import os

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv,cross_entropy,max_margin_loss
from datafree.utils import ImagePool, DataIter, clip_images
import collections
from torchvision import transforms
from kornia import augmentation
from tqdm import tqdm

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

def reset_model(model, reset=1):
    if reset == 0:
        print("generator no reset")
        return
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.iscalloss=False

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization


        if not self.iscalloss:
            return

            
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


    def calloss(self,iscalloss):

        self.iscalloss=iscalloss

class SkyuFeatureLabelImagePool0809(object):

    def __init__(self, root, name='feature_label_images.lmdb'):

        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        lmdb_path= os.path.join(self.root, name)

        self.db = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
            map_size=1099511627776 * 2, readonly=False,
            meminit=False, map_async=True, lock=False)    


        db=self.db

        with db.begin(write=False) as txn:
            stat_dic=txn.stat()
            n_entries=stat_dic['entries']

        if n_entries>0:
            with db.begin(write=False) as txn:
                self.length =pickle5.loads(txn.get(b'__len__'))
                self.keys= pickle5.loads(txn.get(b'__keys__'))
                self._idx= pickle5.loads(txn.get(b'_idx'))
        else:
            self._idx = 0
            self.keys=[]
            self.length=0

        self.lmdb_path=lmdb_path


        savedir_path= os.path.join(self.root, name.split('.')[0])
        self.saveimg_path= os.path.join(savedir_path, 'images')
        self.savefeature_path= os.path.join(savedir_path, 'features')

        os.makedirs(self.saveimg_path, exist_ok=True)
        os.makedirs(self.savefeature_path, exist_ok=True)

        self.dataset=None


    def lmdb_save_feature_label_image_batch(self,features, labels, images, saveprefix):


        db=self.db

        txn=db.begin(write=True)


        for idx, fea in enumerate(features):
            savename = saveprefix+'-%d'%(idx)
            

            txn.put(savename.encode('ascii'), pickle5.dumps(   labels[idx]   ,  protocol=5)  )


            savename_np=  savename+'.npy'

            savename_np_img_loc=  os.path.join(self.saveimg_path, savename_np)
            np.save(savename_np_img_loc, images[idx])

            savename_np_feature_loc=  os.path.join(self.savefeature_path, savename_np)
            np.save(savename_np_feature_loc, features[idx])


            self.length+=1
            self.keys.append(savename)
        txn.commit()

        self._idx+=1


        assert(self.length==len(self.keys))

        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle5.dumps(self.keys,  protocol=5))
            txn.put(b'__len__', pickle5.dumps(self.length,  protocol=5) )
            txn.put(b'_idx', pickle5.dumps(self._idx,  protocol=5) )

        db.sync()

    def add(self, features, labels, images, targets=None):

        self.lmdb_save_feature_label_image_batch(features, labels, images, "%d"%(self._idx) )
        

    def get_dataset(self):

        if self.dataset is not None:

            del self.dataset 

            gc.collect()

        self.dataset = LMDBFeatureLabelImageDataset0809(db=self.db, saveimg_path = self.saveimg_path,  savefeature_path = self.savefeature_path , do_norm_range= False)

        return self.dataset



class SkyuFeatureLabelImagePool1024(object):

    def __init__(self, root, name='feature_label_images.lmdb',batch_size=256):

        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        self._idx = 0
        self.keys=[]
        self.length=0


        savedir_path= os.path.join(self.root, name.split('.')[0])

        self.dataset=None

        n_epoch = 300

        self.all_labels = [[None] * (batch_size+5) ] * (n_epoch+5)
        self.all_images = [[None] * (batch_size+5) ] * (n_epoch+5)
        self.all_features = [[None] * (batch_size+5) ] * (n_epoch+5)



    def lmdb_save_feature_label_image_batch(self,features, labels, images, epochidx):


        nfeature=len(features)

        for idx in range(nfeature):
            savename = '%d-%d'%(epochidx, idx)
            
            self.all_labels[epochidx][idx] =  labels[idx]


            self.all_images[epochidx][idx] = images[idx]


            self.all_features[epochidx][idx] = features[idx]


            self.length+=1
            self.keys.append(savename)

        self._idx+=1

        

        assert(self.length==len(self.keys))



    def add(self, features, labels, images, targets=None):

        self.lmdb_save_feature_label_image_batch(features, labels, images, self._idx )
        

    def get_dataset(self):


        if self.dataset is not None:

            del self.dataset 

            gc.collect()

        db = {'__keys__':  self.keys,
            '__len__': self.length,
            '_idx': self._idx,
            'labels':self.all_labels, 
            'images':self.all_images, 
            'features':self.all_features }

        self.dataset = LMDBFeatureLabelImageDataset1024(db= db , do_norm_range= False)

        return self.dataset


class Spaceshipnet(BaseSynthesis):
    def __init__(self, teacher,student, generator, nz, num_classes, img_size, glmdb_dir,
                 feature_layers=None, bank_size=40960, n_neg=4096, head_dim=128, init_dataset=None,
                 iterations=100, lr_g=1e-3, lr_h=1e-3, progressive_scale=False,aug_typ='channel_mix',
                 synthesis_batch_size=128, sample_batch_size=128, kd_steps=3,first_bn_multiplier=1,
                 bn=1, oh=1, cr=0.8, cr_T=0.1, alpha=0.1,threshold=-2,adv = 0, softmax=True,
                 save_dir='run/cmi', transform=None, use_gen_mixup_cutmix=True, 
                 gen_mixup_cutmix_p=0.7, autocast=None, use_fp16=False, dataset='cifar10',
                 normalizer=None, device='cpu', distributed=False, reset=1):
        super(Spaceshipnet, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.glmdb_dir = glmdb_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_h = lr_h
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.n_neg = n_neg
        self.bn = bn
        self.oh = oh
        self.adv = adv
        self.alpha = alpha
        self.threshold = threshold
        self.softmax = softmax
        self.reset = reset
        self.aug_typ = aug_typ
        if softmax:
            print("Using softmax for max-margin loss")
        else:
            print("No softmax for max-margin loss")
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.bank_size = bank_size
        self.init_dataset = init_dataset

        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir,dataset=dataset)
        # self.data_pool0 = SkyuFeatureLabelImagePool1024(glmdb_dir,name='feature_label_images0.lmdb',batch_size=synthesis_batch_size)
        # self.data_pool1 = SkyuFeatureLabelImagePool1024(glmdb_dir,name='feature_label_images1.lmdb',batch_size=synthesis_batch_size)
        # self.data_pool2 = SkyuFeatureLabelImagePool1024(glmdb_dir,name='feature_label_images2.lmdb',batch_size=synthesis_batch_size)
        self.data_pool0 = SkyuFeatureLabelImagePool0809(glmdb_dir,name='feature_label_images0.lmdb')
        self.data_pool1 = SkyuFeatureLabelImagePool0809(glmdb_dir,name='feature_label_images1.lmdb')
        self.data_pool2 = SkyuFeatureLabelImagePool0809(glmdb_dir,name='feature_label_images2.lmdb')
        self.transform = transform
        self.use_gen_mixup_cutmix = use_gen_mixup_cutmix
        self.gen_mixup_cutmix_p = gen_mixup_cutmix_p
        self.cr = cr
        self.cr_T = cr_T
        self.cmi_hooks = []
        self.labels = None
        self.kd_steps = kd_steps
        self.dataset = dataset
        self.fc_like_conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1,padding=0, bias=False).to(device)
        self.fc_like_conv.eval()
        self.loss_meanvar_feature_layers = []
        self.fake_data = None
        self.data_iter = None
        self.data_iter0 = None
        self.data_iter1 = None
        self.data_iter2 = None
        self.conv_idx = [0,1,2]
        self.first_bn_multiplier=first_bn_multiplier
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
                self.loss_meanvar_feature_layers.append(DeepInversionFeatureHook(module))
        self.generator = generator.to(device).train()
        self.device = device

        if self.dataset in ['cifar10','cifar100','cinic10']:
            self.aug = transforms.Compose([ 
                    augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                    augmentation.RandomHorizontalFlip(),
                    normalizer,
                ])
        else:
            self.aug = transforms.Compose([ 
                    augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                    normalizer,
                ])
        #self.contrast_loss = ContrastLoss(temperature=self.cr_T, contrast_mode='one')

    def set_alpha(self,alpha):
        self.alpha = alpha

    def generate_input(self,test=False,labels= None):
        if labels is None:
            if not test:
                labels=[]
                for i in range(self.synthesis_batch_size):
                    label= np.random.randint(self.num_classes)
                    labels.append(label)
            else:
                labels=[i for i in range(self.num_classes)]* (self.synthesis_batch_size//self.num_classes) + [i for i in range(self.synthesis_batch_size%self.num_classes)]


        codes=[]
        for label in labels:
            tmp=[0 for i in range(self.num_classes)]
            tmp[label]=1
            codes.append(tmp)

        noise= torch.randn( (self.synthesis_batch_size, self.nz-self.num_classes) ).to(self.device)

        code = torch.FloatTensor(codes).to(self.device).contiguous()

        self.labels=   torch.LongTensor(labels).to(self.device)

        return torch.cat([noise,code],dim=1)

    def channel_mix_feature(self, a, b):


        nc=a.size()[0]
        picks=np.random.choice(nc, nc//2, replace=False)
        se2=set(range(nc))-set(picks)
        unpicks=list(se2)
        cmask1=torch.zeros([nc], device=self.device).scatter_(0, torch.LongTensor(picks).to(self.device), 1)
        cmask2=torch.zeros([nc], device=self.device).scatter_(0, torch.LongTensor(unpicks).to(self.device), 1)
        viewidxs= [nc]+[1 for i in range( len(list(a.size()))-1  ) ]
        aug_feature=a*cmask1.view(*viewidxs)+  b*cmask2.view(*viewidxs)  

        return aug_feature


    def channel_mix_batch(self,feature,labels):


        nfeature=feature.size()[0]

        ansfeature= feature.clone().detach()
        anslabels= labels.clone().detach()

        for i in range(nfeature):

            j=nfeature-1-i

            ansfeature[i]=self.channel_mix_feature(feature[i],feature[j])

            anslabels[i]= (labels[i]+labels[j])/2

        return ansfeature,anslabels

    def train_G(self, targets=None):
        self.teacher.eval()
        self.student.eval()
        best_cost = 0xfffffff
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        self.generate_input(test=True,labels=None)
        # if targets is None:
        #     targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        #     targets = targets.sort()[0] # sort for better visualization
        # targets = targets.to(self.device)

        for mm in self.loss_meanvar_feature_layers:
            mm.calloss(True)

        # reset_model(self.generator, self.reset)

        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.1*self.lr_g)

        for it in tqdm(range(self.iterations)):
            fake_data, conv_outs = self.generator(z)
            inputs_jit = self.aug(fake_data)

            teacher_out = self.teacher(inputs_jit)
            cls_loss = cross_entropy(teacher_out, self.labels)
            loss_bn_meanvar=0

            if self.bn>0:
                rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_meanvar_feature_layers)-1)]
                loss_bn_meanvar = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_meanvar_feature_layers)]) * self.bn

            loss_G = cls_loss + loss_bn_meanvar
            with torch.no_grad():
                if  best_inputs is None or best_cost > loss_G.item():
                    best_cost = loss_G.item()

                    layer1_out,conv1_out,conv2_out = conv_outs

                    best_inputs = fake_data
                    best_conv_out  = [layer1_out.clone().detach(), conv1_out.clone().detach(), conv2_out.clone().detach()]

            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

        self.student.train()
        # save best inputs and reset data iter
        # self.data_pool.add( best_inputs ,target=targets)
        self.data_pool0.add( best_conv_out[0].cpu().numpy(), self.labels.cpu().numpy(), best_inputs.clone().detach().clamp(0, 1).cpu().numpy() )
        self.data_pool1.add( best_conv_out[1].cpu().numpy(), self.labels.cpu().numpy(), best_inputs.clone().detach().clamp(0, 1).cpu().numpy() )
        self.data_pool2.add( best_conv_out[2].cpu().numpy(), self.labels.cpu().numpy(), best_inputs.clone().detach().clamp(0, 1).cpu().numpy() )

        dst0 = self.data_pool0.get_dataset()
        dst1 = self.data_pool1.get_dataset()
        dst2 = self.data_pool2.get_dataset()

        train_sampler=None
        loader0 = torch.utils.data.DataLoader(
            dst0, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)    

        loader1 = torch.utils.data.DataLoader(
            dst1, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)

        loader2 = torch.utils.data.DataLoader(
            dst2, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)

        self.data_iter0 = DataIter(loader0)
        self.data_iter1 = DataIter(loader1)
        self.data_iter2 = DataIter(loader2)

        self.fake_data=  best_inputs.clone().detach()

        return {"synthetic": best_inputs}


    def synthesize(self, targets=None):
        self.train_G()

        for mod in self.loss_meanvar_feature_layers:
            mod.calloss(False)

        for it in range(self.kd_steps):
            conv_idx =  int(np.random.choice(self.conv_idx, 1)[0])
            if conv_idx==0:
                tmp_data = self.data_iter0.next()

            elif conv_idx==1:
                tmp_data = self.data_iter1.next()

            elif conv_idx==2:
                tmp_data = self.data_iter2.next()
            else:
                raise(RuntimeError('no such conv_idx'))

            features= tmp_data['feature']
            labels= tmp_data['label']
            images= tmp_data['image']

            features=features.to(self.device)
            labels=labels.to(self.device)
            images=images.to(self.device)

            with torch.no_grad():
                conv_out_mix = features
                if self.use_gen_mixup_cutmix and np.random.rand()<self.gen_mixup_cutmix_p:
                    if self.aug_typ=='channel_mix':
                        conv_out_mix, mixlabels = self.channel_mix_batch( features   , labels ) 
                    else:
                        raise(RuntimeError('aug_typ is not known'))
                    fake_data = self.generator.generate_using_convout(conv_out_mix, conv_idx)
                else:
                    mixlabels = labels
                    fake_data  = self.generator.generate_using_convout(conv_out_mix, conv_idx)
            self.data_pool.add( fake_data ,target=mixlabels)
        dst = self.data_pool.get_dataset(transform=self.transform,labeled=True)
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
        # self.data_iter = DataIter(loader)
        self.data_iter = loader
            
    def sample(self):
        while True:
            for _, (images,target) in enumerate(self.data_iter):
                yield images,target