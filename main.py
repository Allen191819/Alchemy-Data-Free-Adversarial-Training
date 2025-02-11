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
import torchvision.datasets as datasets
import torchvision.utils as utils
import registry
import datafree

from at.evaluate_robustness import robustness_evaluation_summary

from torch.utils.tensorboard import SummaryWriter
from at.evaluate_robustness import perturb_input,eval_adv_test_blackbox,eval_adv_test_whitebox
# from at.mart import mart_loss
from at.awp import TradesAWP
from torch.cuda.amp import autocast, GradScaler

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

parser = argparse.ArgumentParser(description='Data-free Adversarial Training')

# Data Free
parser.add_argument('--at-method', required=True, choices=['trades', 'awp_trades', 'at', 'mart', 'awp_mart'])
parser.add_argument('--method', default='cmi', choices=['diffusion', 'cmi','mm_loss'])
parser.add_argument('--bn', default=1, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=1, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--cr', default=0.8, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--adv', default=0.5, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--cr_T', default=0.1, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--alpha',default=0.1, type=float, help='scaling factor for max margin loss')
parser.add_argument('--threshold',default=-2, type=float, help='threshold factor for max margin loss')
parser.add_argument('--softmax',action='store_true', help='use softmax for max margin loss')
parser.add_argument('--save_dir', default=f'runs/synthesis_{current_time}', type=str)
parser.add_argument('--log_dir', default=f'runs/log/log_{current_time}', type=str)
parser.add_argument('--generator_reset', default=1, type=int,
                    help='whether to reset the generator, 1 for reset, 0 for no')

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for AT')
parser.add_argument('--lr_decay_milestones', default="60,80,100,120", type=str,
                    help='milestones for learning rate decay')
parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--lr_d', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--lr_z', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=500, type=int, metavar='N',
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
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--distance', default="l_inf",type=str,help="Distance of PGD")
parser.add_argument('--awp-warmup',default=30,type=int,help='Warm up epoch of awp')
parser.add_argument('--awp_gamma',default=0.005,type=float,help='Parameter gamma of awp')
parser.add_argument('--mart_beta',default=6,type=float,help='Parameter beta of mart')

# Attack
parser.add_argument('--attack-epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--attack-num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--attack-step-size', default=0.003, type=float,
                    help='perturb step size')

def main():
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    os.makedirs(args.log_dir,exist_ok=True)
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO,    
    handlers=[ logging.FileHandler(os.path.join(args.log_dir,f'dfat{args.log_tag}.log')), logging.StreamHandler(sys.stdout) ])
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
    args.scaler = GradScaler()
    args.autocast = autocast

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = torch.device(f"cuda:{gpu}")

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, test_dataset, train_transform, val_transform = registry.get_dataset(name=args.dataset, data_root=args.data_root,return_transform=True)

    whole_test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size//4, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    _,test_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset)-1000, 1000])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size//4, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    evaluator = datafree.evaluators.classification_evaluator(test_loader)
    # 设置模型，直接继承老师模型的参数
    model = registry.get_model(args.model, num_classes=num_classes).to(args.gpu)
    ori_model = registry.get_model(args.model, num_classes=num_classes, pretrained=True).to(args.gpu).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    checkpoint = torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')
    model.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')['state_dict'])
    ori_model.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.model), map_location='cpu')['state_dict'])
    
    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.sample_batch_size is None:
        args.sample_batch_size = args.batch_size
    
    if args.method=='cmi':
        Syn = None
        pass
    elif args.method=='mm_loss':
        generator = datafree.models.generator.Generator(nz=512, ngf=64, img_size=32, nc=3)
        Syn = datafree.synthesis.CMISynthesizer(teacher=ori_model,student=model,generator=generator,num_classes=num_classes,nz=512,img_size=(3,32,32),synthesis_batch_size=args.synthesis_batch_size,sample_batch_size=args.sample_batch_size,normalizer=args.normalizer,device=args.gpu,save_dir=args.save_dir,iterations=args.g_steps,alpha=args.alpha,oh=args.oh,bn=args.bn,cr=args.cr,cr_T=args.cr_T,lr_g=args.lr_g,threshold=args.threshold,adv=args.adv,transform=train_transform,softmax=args.softmax,dataset=args.dataset, reset=args.generator_reset)
        # for epoch in range(200):
        #     logging.info(f"Current Epoch:{epoch}/{200} ({epoch*args.synthesis_batch_size}/{200*args.synthesis_batch_size}) Alpha: {args.alpha}")
        #     Syn.synthesize()
    else: raise NotImplementedError

    # syn_dataset  = datasets.ImageFolder(args.save_dir,transform=ori_dataset.transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(syn_dataset, [len(syn_dataset)-10000, 10000])
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     pin_memory=True, )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     pin_memory=True
    # )

    # logging.info(f"Loaded dataset,train sample:{len(train_dataset)}, val sample:{len(val_dataset)}, test sample:{len(test_dataset)}")
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
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
                loc = args.gpu
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

    if args.at_method in ["awp_trades", "awp_mart"]:
        proxy = registry.get_model(args.model, num_classes=num_classes, pretrained=False).to(args.gpu)
        proxy.load_state_dict(checkpoint['state_dict'])
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        args.awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma,epochs=args.epochs,normalizer=args.normalizer)
        logging.info(f"args.awp_adversary.gamma: {args.awp_adversary.gamma}")
    else:
        args.awp_adversary = None

    args.writer = SummaryWriter(log_dir=os.path.join(args.log_dir,f"{args.at_method}_{args.log_tag}_tensorboard_writer_log"))

    ############################################
    # Train Loop
    ############################################
    syn_count = args.start_epoch*2
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch=epoch

        for _ in range((args.batch_size*2)//args.synthesis_batch_size):
            logging.info(f"Current Epoch:{syn_count}/{args.epochs*2} ({syn_count*args.synthesis_batch_size}/{args.epochs*args.synthesis_batch_size*2}) Alpha: {args.alpha} Scheduler step:{epoch}")
            Syn.synthesize()
            train(Syn, model, ori_model, optimizer, epoch, args,syn_count)
            syn_count+=1

        scheduler.step()
        model.eval()
        # if epoch!=0 and epoch%10==0:
        #     train_acc1, train_rob_acc_b = eval_adv_test_blackbox(model_target=model, model_source=ori_model, device=args.gpu,  test_loader=train_loader,labeled=False,epsilon=args.epsilon)
        #     train_acc1, train_rob_acc_w = eval_adv_test_whitebox(model=model, device=args.gpu,  test_loader=train_loader,labeled=True,model_original=ori_model,epsilon=args.epsilon)
        #     logging.info(f"[Train] Acc1:{train_acc1:.2f}% B Acc:{train_rob_acc_b:.2f}% W Acc:{train_rob_acc_w:.2f}%")
        if args.dataset == 'mnist':
            syn_dataset  = datafree.utils._utils.LabeledGrayImageDataset(args.save_dir,transform=train_transform)
        else:
            syn_dataset  = datasets.ImageFolder(args.save_dir,transform=train_transform)
        if(len(syn_dataset)>5000):
           _, val_dataset = torch.utils.data.random_split(syn_dataset, [len(syn_dataset)-1000, 1000])
        else:
            val_dataset = syn_dataset
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            pin_memory=True
        )

        logging.info(f"----------Evaluation At Epoch {epoch}---------------")

        acc1, v_r_acc_b = eval_adv_test_blackbox(model_target=model, model_source=ori_model, device=args.gpu,  test_loader=val_loader,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)
        acc1, v_r_acc_w = eval_adv_test_whitebox(model=model, device=args.gpu,  test_loader=val_loader,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)

        logging.info(f"[Val] Acc1:{acc1:.2f}% B Acc:{v_r_acc_b:.2f}% W Acc:{v_r_acc_w:.2f}%")

        args.writer.add_scalars('Val/Acc', {'Natural Acc':acc1,
                                            'B Robust Acc':v_r_acc_b,
                                            'W Robust Acc':v_r_acc_w}, epoch)

        test_acc1, t_r_acc_b = eval_adv_test_blackbox(model_target=model, model_source=ori_model, device=args.gpu,  test_loader=test_loader,labeled=True,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)
        test_acc1, t_r_acc_w = eval_adv_test_whitebox(model=model, device=args.gpu,  test_loader=test_loader,labeled=True,model_original=ori_model,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)

        logging.info(f"[Test] Acc1:{test_acc1:.2f}% B Acc:{t_r_acc_b:.2f}% W Acc:{t_r_acc_w:.2f}%")

        args.writer.add_scalars('Test/Acc', {'Natural Acc':test_acc1,
                                             'B Robust Acc':t_r_acc_b,
                                             'W Robust Acc':t_r_acc_w}, epoch)

        # Different Attacks Evaluation on sys dataset
        logging.info(f"Different Attacks on Val dataset")
        robustness_evaluation_summary(model=model, model_source=ori_model, dataloader=val_loader,eps=args.attack_epsilon,log_path=os.path.join(args.log_dir,f'dfat{args.log_tag}.log'),transform=normalizer,batch_size=args.batch_size,device=args.gpu,logger=logging)
        
        # Different Attacks Evaluation
        logging.info(f"Different Attacks on Test dataset")
        robustness_evaluation_summary(model=model, model_source=ori_model, dataloader=test_loader,eps=args.attack_epsilon,log_path=os.path.join(args.log_dir,f'dfat{args.log_tag}.log'),transform=normalizer,batch_size=args.batch_size,device=args.gpu,logger=logging)

        if epoch!=0 and epoch % 10 ==0:
            test_acc1, t_r_acc_b = eval_adv_test_blackbox(model_target=model, model_source=ori_model, device=args.gpu,  test_loader=whole_test_loader,labeled=True,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)
            test_acc1, t_r_acc_w = eval_adv_test_whitebox(model=model, device=args.gpu,  test_loader=whole_test_loader,labeled=True,model_original=ori_model,epsilon=args.attack_epsilon,num_steps=args.attack_num_steps,step_size=args.attack_step_size,normalizer=normalizer)

            logging.info(f"At epoch {epoch}, [Whole Test Set] Acc1:{test_acc1:.2f}% B Acc:{t_r_acc_b:.2f}% W Acc:{t_r_acc_w:.2f}%")
        
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


def train(Syn,model,ori_model, optimizer,epoch, args,syn_count):
    total_imgs = 0
    correct_imgs = 0
    sum_natural_loss = 0
    sum_robust_loss = 0
    ori_model.eval()
    model.train()
    sampler = Syn.sample()
    for kd in range(syn_count+2):
        (data,target) = sampler.__next__()
        x_natural = data.to(args.gpu)
        # with args.autocast():
        with torch.no_grad():
            target = ori_model(args.normalizer(x_natural))
        y_predicted = target.max(1).indices
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
        if args.at_method in ["awp_trades", "awp_mart"] and syn_count >= args.awp_warmup:
            awp = args.awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         epoch=epoch,
                                         beta=args.beta)
            args.awp_adversary.perturb(awp)
        
        mart_batch_size = len(x_natural)
        kl = nn.KLDivLoss(reduction='none')

        model.train()
        optimizer.zero_grad()

        # 计算loss_mart

        logits = model(args.normalizer(x_natural))
        logits_adv = model(args.normalizer(x_adv))

        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        
        # print("tmp1", tmp1)
        
        # print("tmp1[:, -1]", tmp1[:, -1] )
        
        # print("y", y)

        new_y = torch.where(tmp1[:, -1] == y_predicted, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, y_predicted) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y_predicted.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / mart_batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss_mart = loss_adv + float(args.mart_beta) * loss_robust

        logging.info(f"loss_adv: {loss_adv}")
        logging.info(f"loss_robust: {loss_robust}")

        # with args.autocost():
        
        
        '''
        # loss function design
        # 限制在对抗条件下预测保持不变
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                             F.softmax(logits, dim=1),
                             reduction='batchmean')
        # loss_robust = F.cross_entropy(logits_adv,logits.max(1)[1])
        # loss_robust = max_margin_loss(logits_adv,target.max(1)[1])
        # calculate natural loss and backprop
        # 限制在自然样本的预测和原始模型保持一致
        loss_natural = F.kl_div(F.log_softmax(logits,dim=1),
                                F.softmax(target,dim=1),
                                reduction='batchmean')
        # loss_natural = F.cross_entropy(logits,target.max(1)[1])
        # loss_natural = loss_kl*(1 - epoch/args.epochs) + loss_ce*(epoch/args.epochs)
        # logging.info(f"Loss Natural: {loss_natural}, Loss Robust: {loss_robust}")
        # logging.info(f"logits: {logits[0]}")
        # logging.info(f"logits_adv: {logits_adv[0]}")
        # logging.info(f"targets: {target[0]}")
        if args.at_method in ['trades','awp_trades']:
            loss = loss_natural + args.beta * loss_robust
            with torch.no_grad():
                sum_robust_loss += loss_robust.item()*x_natural.shape[0]
                sum_natural_loss += loss_natural.item()*x_natural.shape[0]
        else:
            loss = loss_robust
            with torch.no_grad():
                sum_robust_loss += loss_robust.item()*x_natural.shape[0]
        
        '''
        
        # update the parameters at last
        # optimizer.zero_grad()
        # args.scaler.scale(loss).backward()
        loss_mart.backward()
        optimizer.step()
        _ ,pred = logits.max(1)
        _ ,t_pred = target.max(1)
        correct_imgs += (pred == t_pred).sum().item()
        total_imgs += len(target)
        if args.at_method in ["awp_trades", "awp_mart"] and syn_count >= args.awp_warmup:
            args.awp_adversary.restore(awp)
    
    train_acc1 = correct_imgs/total_imgs*100
    natural_loss = sum_natural_loss/total_imgs
    robust_loss = sum_robust_loss/total_imgs

    if args.at_method in ['trades','awp_trades']:
        args.writer.add_scalars('Train/Loss', {'Natural Loss':natural_loss,
                                        'Robust Loss':robust_loss,
                                        'Loss': robust_loss*args.beta+natural_loss}, syn_count)
        current_loss = robust_loss*args.beta+natural_loss
    elif args.at_method in ['mart','awp_mart']:
        args.writer.add_scalars('Train/Loss', {
                                        'Loss': loss_mart}, syn_count)
        current_loss = loss_mart
    else:
        args.writer.add_scalars('Train/Loss', { 'Robust Loss':robust_loss.item() }, syn_count)
        current_loss = robust_loss
    args.writer.add_scalar('Train/Acc',train_acc1,syn_count)
    logging.info(f"[Train] Acc1:{train_acc1:.2f}% Loss: {current_loss:.5f} Lr: {optimizer.param_groups[0]['lr']:.6f}")

   
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()

