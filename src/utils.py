import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import torch


def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore

def color_labeling(lbl, num_targets, colors):
    z, h, w = lbl.shape
    new_lbl = np.zeros([z,h,w,3], dtype=np.uint8)
    for i in range(1, num_targets, 1):
        new_lbl[lbl==i] = colors[i-1]
    return new_lbl

def convert_to_rgb(img):
    img = img[...,np.newaxis]
    img = np.concatenate([img, img, img], axis=3)
    return img

def convert_to_8bit(img, p_min=0, p_max=100):
    lower, upper = np.percentile(img, (p_min, p_max))
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    img = (img * 255).astype(np.uint8)
    return img

def create_animation(ims):
    rc('animation', html='jshtml')
    fig = plt.figure(figsize=(6, 3))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")
    text = plt.text(0.05, 0.05, f'index {1}', transform=fig.transFigure, fontsize=16, color='darkblue')
    def animate_func(i):
        im.set_array(ims[i])
        text.set_text(f'index {i+1}')
        return [im]
    plt.close()
    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//10)