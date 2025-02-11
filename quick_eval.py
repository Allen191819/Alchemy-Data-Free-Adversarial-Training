from torchvision import datasets, transforms as T
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
from torch.autograd import Variable
from tqdm import tqdm
from at.evaluate_robustness import robustness_evaluation_summary
import os
import sys
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import datafree
import registry
import torch.nn.functional as F
import argparse
import datetime

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
parser = argparse.ArgumentParser(description='Data-free Adversarial Training')

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--log_dir', default=f'runs/log/log_{current_time}', type=str)
parser.add_argument('--log_tag', default=None)

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# PGD
parser.add_argument('--epsilon', default=0.031,type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,type=float,
                    help='perturb step size')


def main():
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    if args.log_tag is None:
        args.log_tag = f"{args.dataset}_{args.model}"
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO,    
    handlers=[ logging.FileHandler(os.path.join(args.log_dir,f'eval_{args.log_tag}.log')), logging.StreamHandler(sys.stdout) ])
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logging.info(message)
 
    ############################################
    # Setup dataset
    ############################################
    args.gpu = torch.device(f"cuda:{args.gpu}")
    num_classes, ori_dataset, test_dataset, train_transform, _ = registry.get_dataset(name=args.dataset, data_root=args.data_root,return_transform=True)
    normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])

    if len(test_dataset)>20000:
        _, test_dataset = torch.utils.data.random_split(test_dataset,[len(test_dataset)-20000,20000])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    model = registry.get_model(args.model, num_classes=num_classes).to(args.gpu)
    ori_model = registry.get_model(args.model, num_classes=num_classes, pretrained=True).to(args.gpu).eval()
    if args.checkpoint is None:
        args.checkpoint = 'checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['state_dict'])
    ori_model.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')['state_dict'])
    acc_1,  _,_,_ = eval_model(ori_model,test_loader,normalizer,args.gpu)
    # acc_r_1,_,_,_ = eval_attack(model,test_loader,normalizer,args.gpu,args)
    logging.info(f"Pretrained Model => Natural Acc:{acc_1*100:.2f}%")
    # print(f"Pretrained Model => Natural Acc:{acc_1*100:.2f}% Robust Acc:{acc_r_1*100:.2f}%")

    acc_2,  _,_,_ = eval_model(model,test_loader,normalizer,args.gpu)
    # acc_r_2,_,_,_ = eval_attack(ori_model,test_loader,normalizer,args.gpu,args)
    logging.info(f"Robust Model     => Natural Acc:{acc_2*100:.2f}%")
    # print(f"Robust Model     => Natural Acc:{acc_2*100:.2f}% Robust Acc:{acc_r_2*100:.2f}%")

    robustness_evaluation_summary(model=model, model_source=ori_model, dataloader=test_loader,eps=args.epsilon,log_path=os.path.join(args.log_dir,f'eval_{args.log_tag}.log'),transform=normalizer,batch_size=args.batch_size,device=args.gpu,logger=logging)

def print_state_dict(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def print_state_dict_from_checkpoint(checkpoint):
    print("Model's state_dict:")
    for param_tensor in checkpoint["state_dict"]:
        print(param_tensor, "\t", checkpoint["state_dict"][param_tensor].size())



def pgd_whitebox(model,
                  X,
                  y,
                  normalizer,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.003,
                  random=True,
                  device=None):
    X = X.to(device)
    X_pgd = Variable(X.data, requires_grad=True).to(device)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(normalizer(X_pgd)), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    # err_pgd = (model(normalizer(X_pgd)).data.max(1)[1] != y).float().sum()
    # print(f"Err sample: {err_pgd}/{X.shape[0]}")
    return X_pgd

def eval_model(model, dataloader,transform,device):
    total_imgs = 0
    correct_imgs = 0
    total_loss = 0
    min_logict = 0xffffff
    outputs = []
    labels = []
    model.eval()
    for i, (images, target) in enumerate(dataloader):
        total_imgs += len(target)
        images = images.to(device)
        target = target.to(device)
        with torch.no_grad():
            if transform is not None:
                output = model(transform(images))
            else:
                output = model(images)
            loss = F.cross_entropy(output, target)
        outputs.append(output)
        labels.append(target)
        _ ,pred = output.max(1)
        # for index in range(output.shape[0]):
        #     if pred[index]!=target[index]:
        #         continue
        #     second_max = -0xffff
        #     for cls in range(output[index].shape[0]):
        #         if cls == pred[index]:
        #             continue
        #         else:
        #             if output[index][cls]>second_max:
        #                 second_max=output[index][cls]
        #     if second_max<min_logict:
        #         min_logict = second_max
        correct_imgs += (pred == target).sum().item()
        total_loss += loss
    # print(f"Min Second Max logict:{min_logict}")
    return correct_imgs/total_imgs, total_loss/total_imgs, torch.cat(outputs,dim=0), torch.cat(labels,dim=0)

def eval_attack(model, dataloader,transform,device,args):
    total_imgs = 0
    correct_imgs = 0
    total_loss = 0
    outputs = []
    labels = []
    model.eval()
    for i, (images, target) in tqdm(enumerate(dataloader)):
        total_imgs += len(target)
        images = images.to(device)
        target = target.to(device)
        images_adv = pgd_whitebox(model,images,target,transform,
                                  device=device,
                                  epsilon=args.epsilon,
                                  num_steps=args.num_steps,
                                  step_size=args.step_size)
        with torch.no_grad():
            if transform is not None:
                output = model(transform(images_adv))
            else:
                output = model(images_adv)
            loss = F.cross_entropy(output, target)
        outputs.append(output)
        labels.append(target)
        _ ,pred = output.max(1)
        correct_imgs += (pred == target).sum().item()
        total_loss += loss
    return correct_imgs/total_imgs, total_loss/total_imgs, torch.cat(outputs,dim=0), torch.cat(labels,dim=0)

if __name__ == "__main__":
    main()