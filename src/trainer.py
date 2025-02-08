import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

from src.losses import criterion_seg
from src.metrics import fbeta_score_multiclass


def mixup(input, mask, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_mask = mask[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, mask, shuffled_mask, lam

def mixup_dann(input, mask, source, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_mask = mask[indices]
    shuffled_source = source[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, mask, source, shuffled_mask, shuffled_source, lam


def train_func(model, loader_train, optimizer,
               alpha, beta, loss_weights,
               p_mixup, n_epochs_mixup, epoch,
               device, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for data in bar:
        optimizer.zero_grad()
        images = data['image']
        masks = data['label']
        images = images.to(device)
        masks = masks.to(device)

        do_mixup = False
        if (random.random() < p_mixup)&(n_epochs_mixup>epoch):
            do_mixup = True
            images, masks, masks_mix, lam = mixup(images, masks)

        with amp.autocast():
            logits = model(images)
            loss, _, _ = criterion_seg(logits, masks, alpha, beta, loss_weights)
            if do_mixup:
                loss11, _, _ = criterion_seg(logits, masks_mix, alpha, beta, loss_weights)
                loss = loss * lam  + loss11 * (1 - lam)
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)

def train_func_dann(model, loader_train, optimizer,
               alpha, beta, loss_weights,
               p_mixup, n_epochs_mixup, epoch, n_epochs,
               device, scaler=None):
    model.train()
    loss_weights_seg = [loss_weights[0], loss_weights[1]]
    loss_weights_domain = loss_weights[2]
    train_loss = []
    bar = tqdm(enumerate(loader_train), total=len(loader_train))
    for step, data in bar:
        optimizer.zero_grad()
        images = data['image']
        masks = data['label']
        sources = data['source']
        images = images.to(device)
        masks = masks.to(device)
        sources = sources.to(device)
        total_steps = n_epochs * len(bar)
        loader_batch_step = step
        p = float(loader_batch_step + epoch * len(bar)) / total_steps
        a_dann = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        do_mixup = False
        if (random.random() < p_mixup)&(n_epochs_mixup>epoch):
            do_mixup = True
            images, masks, sources, masks_mix, sources_mix, lam = mixup_dann(images, masks, sources)

        with amp.autocast():
            logits_seg, logits_domain = model(images, a_dann)
            loss_seg, _, _ = criterion_seg(logits_seg, masks, alpha, beta, loss_weights_seg)
            loss_domain = F.binary_cross_entropy_with_logits(logits_domain, sources)
            loss = loss_seg + loss_weights_domain * loss_domain
            if do_mixup:
                loss11_seg, _, _ = criterion_seg(logits_seg, masks_mix, alpha, beta, loss_weights_seg)
                loss11_domain = F.binary_cross_entropy_with_logits(logits_domain, sources_mix)
                loss11 = loss11_seg + loss_weights_domain * loss11_domain
                loss = loss * lam  + loss11 * (1 - lam)
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid, alpha, beta, loss_weights, device):
    model.eval()
    valid_loss, valid_loss_bce, valid_loss_tvs = [], [], []
    gts, outputs = [], []
    bar = tqdm(enumerate(loader_valid), total=len(loader_valid))
    with torch.no_grad():
        for step, data in bar:
            images = data['image']
            masks = data['label']
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            gts.append(masks.cpu().detach().numpy())
            outputs.append(logits.cpu().detach().numpy())
            loss, loss_bce, loss_tvs = criterion_seg(logits, masks, alpha, beta, loss_weights)
            valid_loss.append(loss.item())
            valid_loss_bce.append(loss_bce.item())
            valid_loss_tvs.append(loss_tvs.item())

            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
    gts = np.concatenate(gts)
    outputs = np.concatenate(outputs)
    outputs = outputs.argmax(1)
    score = fbeta_score_multiclass(gts, outputs, beta=4.0, num_classes=7)
    return np.mean(valid_loss), np.mean(valid_loss_bce), np.mean(valid_loss_tvs), score


def valid_func_dann(model, loader_valid, alpha, beta, loss_weights, device):
    model.eval()
    loss_weights_seg = [loss_weights[0], loss_weights[1]]
    loss_weights_domain = loss_weights[2]
    valid_loss, valid_loss_bce_seg, valid_loss_tvs_seg, valid_loss_domain = [], [], [], []
    gts, outputs = [], []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for data in bar:
            images = data['image']
            masks = data['label']
            sources = data['source']
            images = images.to(device)
            masks = masks.to(device)
            sources = sources.to(device)

            logits_seg, logits_domain = model(images, 1)
            gts.append(masks.cpu().detach().numpy())
            outputs.append(logits_seg.cpu().detach().numpy())
            loss_seg, loss_bce_seg, loss_tvs_seg = criterion_seg(logits_seg, masks, alpha, beta, loss_weights_seg)
            loss_domain = F.binary_cross_entropy_with_logits(logits_domain, sources)
            loss = loss_seg + loss_weights_domain * loss_domain
            valid_loss.append(loss.item())
            valid_loss_bce_seg.append(loss_bce_seg.item())
            valid_loss_tvs_seg.append(loss_tvs_seg.item())
            valid_loss_domain.append(loss_domain.item())

            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
    gts = np.concatenate(gts)
    outputs = np.concatenate(outputs)
    outputs = outputs.argmax(1)
    score = fbeta_score_multiclass(gts, outputs, beta=4.0, num_classes=7)
    return np.mean(valid_loss), np.mean(valid_loss_bce_seg), np.mean(valid_loss_tvs_seg),\
     np.mean(valid_loss_domain), score


def valid_func2(model, loader_valid, alpha, beta, loss_weights, device):
    model.eval()
    valid_loss, valid_loss_bce, valid_loss_tvs = [], [], []
    bar = tqdm(loader_valid)
    scores = []
    with torch.no_grad():
        for data in bar:
            images = data['image']
            masks = data['label']
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss, loss_bce, loss_tvs = criterion_seg(logits, masks, alpha, beta, loss_weights)
            outputs = logits.argmax(1)
            score_ = fbeta_score_multiclass(masks.cpu().detach().numpy(), outputs.cpu().detach().numpy(), beta=4.0, num_classes=7)
            scores.append(score_)
            valid_loss.append(loss.item())
            valid_loss_bce.append(loss_bce.item())
            valid_loss_tvs.append(loss_tvs.item())

            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
    
    return np.mean(valid_loss), np.mean(valid_loss_bce), np.mean(valid_loss_tvs), np.mean(scores)



def inf_func(model, loader_valid, size, device):
    model.eval()
    gts, outputs = [], []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for data in bar:
            images = data['image']
            masks = data['label']
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            gts.append(masks.cpu())
            outputs.append(logits.cpu())

    gts = torch.concat(gts)
    outputs = torch.concat(outputs)
    print(gts.shape, outputs.shape)
    outputs = outputs.permute(1, 0, 2, 3)  # [channels, depth, height, width]
    # リサイズのために形状を変更
    #outputs = outputs.reshape(channels * depth, 1, height, width)  # [channels * depth, 1, height, width]
    outputs = F.interpolate(outputs.unsqueeze(0), size=size, mode='trilinear', align_corners=False).squeeze(0)  # [channels * depth, 1, new_height, new_width]
    #outputs = outputs.reshape(channels, depth, size, size)  # [channels, depth, new_height, new_width]

    # masksをリサイズ
    # [batch=depth, height, width] -> [depth, new_height, new_width]
    gts = F.interpolate(gts.unsqueeze(0).unsqueeze(0).float(), size=size, mode='nearest').squeeze(0).squeeze(0).long()  # [depth, new_height, new_width]

    return gts, outputs





def inf_func_dann(model, loader_valid, size, device):
    model.eval()
    gts, outputs = [], []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for data in bar:
            images = data['image']
            masks = data['label']
            images = images.to(device)
            masks = masks.to(device)

            logits_seg, logits_domain = model(images, 1)
            gts.append(masks.cpu())
            outputs.append(logits_seg.cpu())

    gts = torch.concat(gts)
    outputs = torch.concat(outputs)
    print(gts.shape, outputs.shape)
    outputs = outputs.permute(1, 0, 2, 3)  # [channels, depth, height, width]
    # リサイズのために形状を変更
    #outputs = outputs.reshape(channels * depth, 1, height, width)  # [channels * depth, 1, height, width]
    outputs = F.interpolate(outputs.unsqueeze(0), size=size, mode='trilinear', align_corners=False).squeeze(0)  # [channels * depth, 1, new_height, new_width]
    #outputs = outputs.reshape(channels, depth, size, size)  # [channels, depth, new_height, new_width]

    # masksをリサイズ
    # [batch=depth, height, width] -> [depth, new_height, new_width]
    gts = F.interpolate(gts.unsqueeze(0).unsqueeze(0).float(), size=size, mode='nearest').squeeze(0).squeeze(0).long()  # [depth, new_height, new_width]

    return gts, outputs