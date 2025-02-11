import argparse
import os
import random
import sys
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils

from math import gamma
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from at.evaluate_robustness import perturb_input,eval_adv_test_blackbox,eval_adv_test_whitebox
from at.awp import TradesAWP

import registry
import datafree


# Normal AT
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

parser = argparse.ArgumentParser(description='Data-free Adversarial Training')

parser.add_argument('--at-method', required=True, choices=['trades', 'awp_trades', 'at'])
parser.add_argument('--save_dir', default=None, type=str)
parser.add_argument('--log_dir', default=f'runs/log/log_{current_time}', type=str)

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for AT')
parser.add_argument('--lr_decay_milestones', default="60,100,140,160", type=str,
                    help='milestones for learning rate decay')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--synthesis_epochs', default=200, type=int, metavar='N',
                    help='number of total synthesis epochs to run')
parser.add_argument('--g_steps', default=400, type=int, metavar='N',
                    help='number of iterations for generation')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--sample_batch_size', default=512, type=int)

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# Trades loss && AWP
parser.add_argument('--epsilon', default=0.031,type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--distance', default="l_inf",type=str,help="Distance of PGD")
parser.add_argument('--awp-warmup',default=10,type=int,help='Warm up epoch of awp')
parser.add_argument('--awp-gamma',default=0.005,type=float,help='Parameter gamma of awp')

# Attack
parser.add_argument('--attack-epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--attack-num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--attack-step-size', default=0.003, type=float,
                    help='perturb step size')

def main():
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO,    
    handlers=[ logging.FileHandler(os.path.join(args.log_dir,f'at_{args.log_tag}.log')), logging.StreamHandler(sys.stdout) ])
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logging.info(message)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        logging.warning('You have chosen a specific GPU. This will completely '
                         'disable data parallelism.')
        logging.info(f"Current device: GPU:{args.gpu} for training.")
    else:
        logging.info(f"Current device: CPU for training.")

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = torch.device(f"cuda:{gpu}")

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, test_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)

    if len(test_dataset)>10000:
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [10000, len(test_dataset) - 10000])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(test_loader)
    model = registry.get_model(args.model, num_classes=num_classes).to(args.gpu)
    ori_model = registry.get_model(args.model, num_classes=num_classes, pretrained=True).to(args.gpu).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    checkpoint = torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')
    model.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')['state_dict'])
    ori_model.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')['state_dict'])
    
    if args.save_dir is not None:
        if args.dataset == 'mnist':
            syn_dataset  = datafree.utils._utils.LabeledGrayImageDataset(args.save_dir,transform=ori_dataset.transform)
        else:
            syn_dataset  = datasets.ImageFolder(args.save_dir,transform=ori_dataset.transform)
    else:
        syn_dataset = ori_dataset

    train_dataset, val_dataset = torch.utils.data.random_split(syn_dataset, [len(syn_dataset)-5000, 5000])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True, )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        pin_memory=True
    )

    logging.info(f"Loaded dataset,train sample:{len(train_dataset)}, val sample:{len(val_dataset)}, test sample:{len(test_dataset)}")
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(model, nn.Module):
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.module.load_state_dict(checkpoint['state_dict'])
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: 
                logging.info("Fails to load additional model information")
        else:
            logging.info("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        model.eval()
        eval_results = evaluator(model, device=args.gpu)
        logging.info('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    if args.at_method == "awp_trades":
        proxy = registry.get_model(args.model, num_classes=num_classes, pretrained=False).to(args.gpu)
        proxy.load_state_dict(checkpoint['state_dict'])
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        args.awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma,epochs=args.epochs,normalizer=args.normalizer)
    else:
        args.awp_adversary = None

    args.writer = SummaryWriter(log_dir=os.path.join(args.log_dir,f"{args.at_method}_tensorboard_writer_log"))

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch=epoch
        logging.info(f"Current epoch :{epoch} ......")
        train(train_loader, model, optimizer,epoch, args)
        scheduler.step()
        model.eval()

        acc1, v_r_acc_b = eval_adv_test_blackbox(model_target=model, model_source=ori_model, device=args.gpu,  test_loader=val_loader,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)
        acc1, v_r_acc_w = eval_adv_test_whitebox(model=model, device=args.gpu,  test_loader=val_loader,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)

        logging.info(f"[Val] Acc1:{acc1:.2f}% B Acc:{v_r_acc_b:.2f}% W Acc:{v_r_acc_w:.2f}%")

        args.writer.add_scalars('Val/Acc', {'Natural Acc':acc1,
                                            'B Robust Acc':v_r_acc_b,
                                            'W Robust Acc':v_r_acc_w}, epoch)

        test_acc1, t_r_acc_b = eval_adv_test_blackbox(model_target=model, model_source=ori_model, device=args.gpu,  test_loader=test_loader,labeled=True,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)
        test_acc1, t_r_acc_w = eval_adv_test_whitebox(model=model, device=args.gpu,  test_loader=test_loader,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)

        logging.info(f"[Test] Acc1:{test_acc1:.2f}% B Acc:{t_r_acc_b:.2f}% W Acc:{t_r_acc_w:.2f}%")

        args.writer.add_scalars('Test/Acc', {'Natural Acc':test_acc1,
                                             'B Robust Acc':t_r_acc_b,
                                             'W Robust Acc':t_r_acc_w}, epoch)

        if epoch%5==0:
            ckpt = os.path.join(args.log_dir,'%s_%s_lr%s_%s.pth'%(args.dataset, args.model, args.lr,current_time))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, ckpt)
            logging.info(f'Save model to {ckpt} at epoch:{epoch}')



def train(dataloader,model, optimizer,epoch, args):
    total_imgs = 0
    correct_imgs = 0
    sum_natural_loss = 0
    sum_robust_loss = 0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader),desc=f"Epoch {epoch}"):
        x_natural = data.to(args.gpu)
        target = target.to(args.gpu)
        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              step_size=args.step_size,
                              epsilon=args.epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.distance,
                              normalizer=args.normalizer,
                              device=args.gpu)

        if False:
            with torch.no_grad():
                samples = x_natural.data.cpu()[:64]
                grid = utils.make_grid(samples)
                args.writer.add_image(f"nat-{batch_idx}",grid,batch_idx)
                samples = x_adv.data.cpu()[:64]
                grid = utils.make_grid(samples)
                args.writer.add_image(f"adv-{batch_idx}",grid,batch_idx)
        # calculate adversarial weight perturbation
        if args.at_method == "awp_trades" and epoch >= args.awp_warmup:
            awp = args.awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         epoch=epoch,
                                         datafree=args.save_dir is not None,
                                         beta=args.beta)
            args.awp_adversary.perturb(awp)
        model.train()
        optimizer.zero_grad()
        logits = model(args.normalizer(x_natural))
        logits_adv = model(args.normalizer(x_adv))
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits, dim=1),
                               reduction='batchmean')
        # loss_robust = F.cross_entropy(logits_adv,target)
        # calculate natural loss and backprop

        if args.save_dir is not None:
            loss_natural = F.kl_div(F.log_softmax(logits,dim=1),
                                    F.softmax(target,dim=1),
                                    reduction='batchmean')
        else:
            loss_natural = F.cross_entropy(logits,target)
        if args.at_method in ['trades','awp_trades']:
            loss = loss_natural + args.beta * loss_robust
            with torch.no_grad():
                sum_robust_loss += loss_robust.item()*x_natural.shape[0]
                sum_natural_loss += loss_natural.item()*x_natural.shape[0]
        else:
            loss = loss_robust
            with torch.no_grad():
                sum_robust_loss += loss_robust.item()*x_natural.shape[0]

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _ ,pred = logits.max(1)
        if args.save_dir is not None:
            _ ,t_pred = target.max(1)
        else:
            t_pred = target
        correct_imgs += (pred == t_pred).sum().item()
        total_imgs += len(target)
        if args.at_method == "awp_trades" and epoch >= args.awp_warmup:
            args.awp_adversary.restore(awp)

    train_acc1 = correct_imgs/total_imgs*100
    natural_loss = sum_natural_loss/total_imgs
    robust_loss = sum_robust_loss/total_imgs

    if args.at_method in ['trades','awp_trades']:
        args.writer.add_scalars('Train/Loss', {'Natural Loss':natural_loss,
                                        'Robust Loss':robust_loss,
                                        'Loss': robust_loss*args.beta+natural_loss}, epoch)
        current_loss = robust_loss*args.beta+natural_loss
    else:
        args.writer.add_scalars('Train/Loss', { 'Robust Loss':robust_loss.item() }, epoch)
        current_loss = robust_loss
    args.writer.add_scalar('Train/Acc',train_acc1,epoch)
    logging.info(f"[Train] Acc1:{train_acc1:.2f}% Loss: {current_loss} Lr: {optimizer.param_groups[0]['lr']}")

   
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()
