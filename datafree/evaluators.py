from tqdm import tqdm
import torch.nn.functional as F 
import torch
from . import metrics
from .utils import Normalizer

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    'cub200':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_cars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'tiny_imagenet': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    
    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


class Evaluator(object):
    def __init__(self, metric, dataloader,dataset='cifar10'):
        self.normalizer = Normalizer(**NORMALIZE_DICT[dataset])
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(self.normalizer(inputs))
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

class AdvEvaluator(object):
    def __init__(self, metric, dataloader, adversary):
        self.dataloader = dataloader
        self.metric = metric
        self.adversary = adversary

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = self.adversary.perturb(inputs, targets)
            with torch.no_grad():
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader,dataset="cifar10"):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader,dataset=dataset)

def advarsarial_classification_evaluator(dataloader, adversary):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return AdvEvaluator( metric, dataloader=dataloader, adversary=adversary)


def segmentation_evaluator(dataloader, num_classes, ignore_idx=255):
    cm = metrics.ConfusionMatrix(num_classes, ignore_idx=ignore_idx)
    metric = metrics.MetricCompose({
        'mIoU': metrics.mIoU(cm),
        'Acc': metrics.Accuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)