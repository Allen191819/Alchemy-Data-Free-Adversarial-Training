import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import registry
import datafree
from at import attacks
import torch.optim as optim
from datafree.criterions import max_margin_loss
from torch.autograd import Variable
from torchvision import datasets, transforms
from autoattack import AutoAttack
from tqdm import tqdm

def robustness_evaluation_summary(model, model_source, dataloader, eps, log_path ,batch_size , transform=None,logger=None ,device="cpu"):
    if transform is not None:
        model_ = lambda x: model(transform(x))
        model_source_ = lambda x: model_source(transform(x))
    else:
        model_ = model
        model_source_ = model_source
    criterion = nn.CrossEntropyLoss()
    attacks_blackbox = {
        # "fgsm_attack" : attacks.create_attack(model=model_source_,criterion=criterion,attack_type="fgsm",attack_eps=eps,device=device),
        # "pgd_10_attack" : attacks.create_attack(model=model_source_,criterion=criterion, attack_type="linf-pgd",attack_eps=eps,attack_iter=10,attack_step=(eps/10.0)*2,rand_init_type="uniform",device=device),
        "pgd_20_attack" : attacks.create_attack(model=model_source_,criterion=criterion, attack_type="linf-pgd",attack_eps=eps,attack_iter=20,attack_step=(eps/20.0)*2,rand_init_type="uniform",device=device),
        "apgd_attack" : attacks.create_attack(model=model_source_,criterion="ce", attack_type="linf-apgd",attack_eps=eps, attack_iter=20,device=device),
        "fab_attack" : attacks.create_attack(model=model_source_,criterion="ce",attack_type="fab", attack_eps=eps, attack_iter=50,device=device),
        # "deepfool_attack" :  attacks.create_attack(model=model_source_,criterion=criterion,attack_type="linf-df",attack_eps=eps,attack_iter=50,device=device),
        # "spsa_attack" : attacks.create_attack(model=model_source_,criterion=criterion, attack_type="linf-spsa", attack_eps=eps, attack_iter=20,device=device)
    }

    attacks_whitebox = {
        # "fgsm_attack" : attacks.create_attack(model=model_,criterion=criterion,attack_type="fgsm",attack_eps=eps,device=device),
        # "pgd_10_attack" : attacks.create_attack(model=model_,criterion=criterion, attack_type="linf-pgd",attack_eps=eps,attack_iter=10,attack_step=(eps/10.0)*2,rand_init_type="uniform",device=device),
        "pgd_20_attack" : attacks.create_attack(model=model_,criterion=criterion, attack_type="linf-pgd",attack_eps=eps,attack_iter=20,attack_step=(eps/20.0)*2,rand_init_type="uniform",device=device),
        "apgd_attack" : attacks.create_attack(model=model_,criterion="ce", attack_type="linf-apgd",attack_eps=eps, attack_iter=20,device=device),
        "fab_attack" : attacks.create_attack(model=model_,criterion="ce",attack_type="fab", attack_eps=eps, attack_iter=50,device=device),
        # "deepfool_attack" :  attacks.create_attack(model=model_,criterion=criterion,attack_type="linf-df",attack_eps=eps,attack_iter=50,device=device),
        # "spsa_attack" : attacks.create_attack(model=model_,criterion=criterion, attack_type="linf-spsa", attack_eps=eps, attack_iter=20,device=device)
    }

    attacks_acc = {
        "fgsm_attack" : [0,0],
        "pgd_10_attack" : [0,0],
        "pgd_20_attack" : [0,0],
        "apgd_attack" : [0,0],
        "fab_attack" : [0,0],
        "deepfool_attack" : [0,0],
        "spsa_attack" : [0,0] 
    }
    total_imgs = 0
    for x, y in tqdm(dataloader,desc="Eval attacks......"):
        x,y = x.to(device),y.to(device)
        total_imgs += y.shape[0]
        for method, eval_attack in attacks_blackbox.items():
            x_adv = eval_attack.perturb(x,y)[0]
            with torch.no_grad():
                out = model_(x_adv)
            _, pred = out.max(1)
            attacks_acc[method][0] += (pred == y).sum().item()
        for method, eval_attack in attacks_whitebox.items():
            x_adv = eval_attack.perturb(x,y)[0]
            with torch.no_grad():
                out = model_(x_adv)
            _, pred = out.max(1)
            attacks_acc[method][1] += (pred == y).sum().item()
    
    for method, acc in attacks_acc.items():
        if logger is None:
            print(f"Attack => {method}: B Acc: {acc[0]/total_imgs*100:.4f}% W Acc: {acc[1]/total_imgs*100:.4f}%")
        else:
            logger.info(f"Attack => {method}: B Acc: {acc[0]/total_imgs*100:.4f}% W Acc: {acc[1]/total_imgs*100:.4f}%")

    adversary = AutoAttack(model_, norm="Linf", eps=eps, log_path=log_path,device=device)
    l = [x for (x, y) in dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in dataloader]
    y_test = torch.cat(l, 0)
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    del x_test, y_test

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  normalizer,
                  device=None):
    out = model(normalizer(X))
    err = (out.data.max(1)[1] != y.data).float().sum()
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
    err_pgd = (model(normalizer(X_pgd)).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  normalizer,
                  device=None):
    out = model_target(normalizer(X))
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True).to(device)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(normalizer(X_pgd)), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(normalizer(X_pgd)).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader,normalizer,model_original=None,epsilon=0.031, num_steps=20, step_size=0.003, random=True,labeled=True):
    """
    evaluate model by white-box attack
    """
    robust_err_total = 0
    natural_err_total = 0
    data_total = 0
    if model_original is not None:
        model_original.eval()
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        if labeled:
            target = target.to(device)
        else:
            target = model_original(normalizer(data)).max(1)[1]
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model=model, X=X, y=y,epsilon=epsilon,num_steps=num_steps,step_size=step_size,random=random,device=device,normalizer=normalizer)
        robust_err_total += err_robust
        natural_err_total += err_natural
        data_total += data.shape[0]
    return 100 - natural_err_total.item()/data_total*100, 100 - robust_err_total.item()/data_total*100


def eval_adv_test_blackbox(model_target, model_source, device, test_loader,normalizer, epsilon=0.031, num_steps=20, step_size=0.003, random=True, labeled=True):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0
    data_total = 0
    for data, target in test_loader:
        data = data.to(device)
        if labeled:
            target = target.to(device)
        else:
            target = model_source(normalizer(data)).max(1)[1]
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target=model_target, model_source=model_source, X=X, y=y,epsilon=epsilon,num_steps=num_steps,step_size=step_size,random=random,device=device,normalizer=normalizer)
        robust_err_total += err_robust
        natural_err_total += err_natural
        data_total += data.shape[0]
    return 100 - natural_err_total.item()/data_total*100, 100 - robust_err_total.item()/data_total*100


def perturb_input(model,
                  x_natural,
                  normalizer,
                  step_size=0.007,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  device=torch.device("cuda:0")
                  ):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        y_natural = model(normalizer(x_natural))
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # loss_kl = F.kl_div(F.log_softmax(model(normalizer(x_adv)), dim=1),
                #                    F.softmax(y_natural, dim=1),
                #                    reduction='sum')
                # loss_oh = F.cross_entropy(model(normalizer(x_adv,device)),y_natural.max(1)[1],reduction='sum')
                loss_mm = max_margin_loss(model(normalizer(x_adv)),y_natural.max(1)[1])
            grad = torch.autograd.grad(loss_mm, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(normalizer(adv)), dim=1),
                                       F.softmax(model(normalizer(x_natural)), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv