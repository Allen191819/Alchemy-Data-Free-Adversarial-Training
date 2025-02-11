from torchvision import datasets, transforms as T
from torch.autograd import Variable
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import datafree
import registry
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Natural Training')

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--save_dir', default='checkpoints/pretrained')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for AT')
parser.add_argument('--lr_decay_milestones', default="60,100,140,180", type=str,
                    help='milestones for learning rate decay')
# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

def main():
    args = parser.parse_args()
    ############################################
    # Setup dataset
    ############################################
    args.gpu = torch.device(f"cuda:{args.gpu}")
    num_classes, train_dataset, test_dataset, train_transform, _ = registry.get_dataset(name=args.dataset, data_root=args.data_root,return_transform=True)
    normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    model = registry.get_model(args.model, num_classes=num_classes).to(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    ckpt = os.path.join(args.save_dir,'%s_%s.pth'%(args.dataset, args.model))
    for epoch in range(args.epochs):
        train_acc,train_loss = train_model(model=model,dataloader=train_loader,optimizer=optimizer,scheduler=scheduler,transform=normalizer,device=args.gpu)
        print(f"[Train]({epoch}\{args.epochs}) => Acc {train_acc*100:.2f}%, Loss {train_loss:.4f}")
        val_acc, val_loss = eval_model(model=model,dataloader=test_loader,transform=normalizer,device=args.gpu)
        print(f"[Val]({epoch}\{args.epochs})   => Acc {val_acc*100:.2f}%, Loss {val_loss:.4f}")

        if epoch%5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, ckpt)
            print(f'Save model to {ckpt} at epoch:{epoch}')

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def eval_model(model, dataloader,transform,device):
    total_imgs = 0
    correct_imgs = 0
    total_loss = 0
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
        correct_imgs += (pred == target).sum().item()
        total_loss += loss

    return correct_imgs/total_imgs, total_loss/total_imgs

def train_model(model, dataloader, optimizer, scheduler, transform, device):
    total_imgs = 0
    correct_imgs = 0
    total_loss = 0
    model.train()
    for _, (images, target) in enumerate(dataloader):
        total_imgs += len(target)
        images = images.to(device)
        target = target.to(device)
        if transform is not None:
            output = model(transform(images))
        else:
            output = model(images)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _ ,pred = output.max(1)
        correct_imgs += (pred == target).sum().item()
        total_loss += loss
    scheduler.step()
    return correct_imgs/total_imgs, total_loss/total_imgs

if __name__ == "__main__":
    main()