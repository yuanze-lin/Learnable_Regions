from tqdm import tqdm
from einops import rearrange
from PIL import Image
from copy import deepcopy
from torchvision.utils import save_image
from typing import List, Optional, Union
from torch import autocast
from torchvision import utils as vutils
from utils.util import build_dataset, plot_images
from lr_schedule import WarmupLinearLRSchedule
from torch.utils.tensorboard import SummaryWriter
from models.model import RGN
from models.utils import visualize_images, read_image_from_url, draw_image_with_bbox_new, Bbox
from utils.util2 import compose_text_with_templates, get_augmentations_template
from torchvision.utils import draw_bounding_boxes
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from engine import *
from utils.post_process import get_final_img
import random
import os, jax, cv2, pdb
import numpy as np
import argparse, torch, inspect
import PIL, time, json, datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils.misc as misc
import torchvision.transforms as T
import torch.distributed as dist

def map_cooridates(bbox, min_num=0, max_num=255):
    # input feat size: 32 x 32
    min_num2, max_num2 = 0, 31
    return (max_num-min_num)/(max_num2-min_num2) * \
            (bbox-min_num2) + min_num

def get_mask_imgs(imgs, bboxs):
    imgs = imgs.repeat_interleave(bboxs.shape[0]//imgs.shape[0], 0)
    mask_imgs = torch.zeros(imgs.shape, dtype=torch.uint8)
    for i in range(imgs.shape[0]):
        mask_imgs[i][:, bboxs[i][1].int().item():bboxs[i][3].int().item(), \
                bboxs[i][0].int().item():bboxs[i][2].int().item()] = 1
    return imgs, mask_imgs.float()

def save_img(args, batch, results, bboxs, imgs, mask_imgs, editing_rompt, input_caption):
    res = results/255.
    output_dir = args.output_dir
    new_path = os.path.join(output_dir, str(batch)+'_'+input_caption)
    new_path2 = os.path.join(new_path, 'results')
    new_path3 = os.path.join(new_path, 'boxes')

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    if not os.path.exists(new_path2):
        os.mkdir(new_path2)
    if not os.path.exists(new_path3):
        os.mkdir(new_path3)
        
    transform = T.Resize(512)
    for i in range(results.shape[0]):
        img = (imgs[i]*255.0).to(dtype=torch.uint8)
        bbox = bboxs[i].to(dtype=torch.uint8).unsqueeze(0)
        draw_img = draw_bounding_boxes(img, bbox, width=3, colors=(255,255,0))
        
        img_name = '-'.join(str(editing_rompt).split(' '))
        ori_img_path = os.path.join(new_path, 'input_image.png')
        if i == 0:
            save_image(transform(imgs[i]), ori_img_path)
        save_image(res[i], os.path.join(new_path2, str(batch) + '_' +str(img_name) + 'anchor'+ str(i)+'.png'))
        if args.draw_box:
            bbox = bboxs[i].to(dtype=torch.uint8).unsqueeze(0)
            draw_img = draw_bounding_boxes(img, bbox, width=3, colors=(255,255,0))
            draw_img_path = os.path.join(new_path3, str(batch) + '_' + str(img_name) + 'anchor' + str(i)+'_ori_draw.png')
            save_image(transform((draw_img/255.0).float()), draw_img_path)
            
    get_final_img(args, editing_rompt, ori_img_path, new_path2)

def predict(args, model, template, data_loader_test, device_id):
    for data_iter_step, (imgs, o_prompt, e_prompt) in enumerate(data_loader_test):
        imgs = imgs.to(device=device_id, non_blocking=True)[0].unsqueeze(0)
        o_prompt, e_prompt = o_prompt[0], e_prompt[0]
        e_prompt = compose_text_with_templates(e_prompt, template)
        bboxs = torch.ceil(map_cooridates(model.module.get_anchor_box(imgs)))
        imgs = imgs.repeat_interleave(bboxs.shape[0]//imgs.shape[0], 0)
        _, mask_imgs = get_mask_imgs(imgs, bboxs)
        results = model.module.generate_result(imgs, mask_imgs.to(device_id), e_prompt)
        save_img(args, data_iter_step, results, bboxs, imgs, mask_imgs, e_prompt, o_prompt)

