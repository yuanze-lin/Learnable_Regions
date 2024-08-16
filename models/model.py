from typing import Tuple
from torchvision.utils import save_image
from torchvision import transforms
from einops import rearrange
from PIL import ImageFilter, Image
from torchvision.ops import roi_pool, roi_align
from engine import *
from models.clip_extractor import ClipExtractor
from utils.util2 import get_augmentations_template, spherical_dist_loss, cosine_loss
from models.utils import Bbox

import os
import torch.nn as nn
import numpy as np
import torch
import pdb
import torch.nn.functional as F
import models.vision_transformer as vits
import torch.distributed as dist

class RGN(nn.Module):
    def __init__(self, image_size, device, args):
        super(RGN, self).__init__()
        self.device = device

        # Define DINO
        self.alpha, self.beta, self.gamma = args.loss_alpha, args.loss_beta, args.loss_gamma
        self.threshold, patch_size = 0.65, 8
        self.checkpoint_path = args.load_checkpoint_path
        self.sample_number = args.point_number
        self.rank = dist.get_rank()
        self.max_window_size = args.max_window_size
        self.pipe, self.generator = init_diffusion_engine(args.diffusion_model_path, device)
        self.dino = vits.__dict__["vit_base"](patch_size=patch_size, num_classes=0).to(device)
        self.dino.eval()
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth")
        self.dino.load_state_dict(state_dict, strict=True)
        
        emb_dim, emb_dim2 = 12,8

        self.box = torch.Tensor(range(4, self.max_window_size+1, 2)).to(device)
        # Define AnchorNet
        self.anchor_net = nn.Sequential(
                nn.Conv2d(len(self.box)*emb_dim, emb_dim2, 1),
                nn.ReLU(),
                nn.Conv2d(emb_dim2, 4, 1),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(16*16, 4),
                nn.ReLU(),
                nn.Linear(4, len(self.box)),
                nn.LogSoftmax(dim=1)
                )
        self.conv = nn.Conv2d(768, emb_dim, 1, stride=1)
        self.anchor_net = self.anchor_net.to(device)
        
        self.load_checkpoints()
        
        # define CLIP
        self.clip_extractor = ClipExtractor(self.device)
        self.text_criterion = cosine_loss

    def load_checkpoints(self):
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location='cuda:0')
                new_dict = {}
                for key, value in ckpt.items():
                    if 'module.anchor_net.' in key:
                        key = key[18:]
                        new_dict[key] = value
                self.anchor_net.load_state_dict(new_dict, strict=True)
                if self.rank == 0:
                    print('Checkpoint loaded successfully!')
            except:
                if self.rank == 0:
                    print('Failed to load checkpoint!')

    def sample_point(self, arr, size=1):
        sample_points = []
        for i in range(arr.shape[0]):
            value, index = torch.topk(arr[i].flatten(), len(arr[i].flatten()))
            arr_points = np.array(np.unravel_index(index.cpu().numpy(), arr[i].shape)).T
            arr_points_thre = arr_points[np.where((0<= arr_points[:, 0]-self.max_window_size) & \
                        (0<= arr_points[:, 1]-self.max_window_size) & (arr_points[:, 0] + \
                        self.max_window_size<=31 ) & (arr_points[:, 1] + self.max_window_size <=31))]
            points = torch.tensor(arr_points_thre[:self.sample_number])
            if self.sample_number > len(arr_points_thre):
                remain = self.sample_number-len(arr_points_thre)
                points = torch.cat((points, torch.tensor(arr_points[:remain])), dim=0)
            
            sample_points.append(points.unsqueeze(0))
        sample_points = torch.cat(sample_points, dim=0)
        
        return sample_points
    
    def get_anchor_box(self, imgs):
        w_featmap, h_featmap = 32, 32
        bs = imgs.shape[0]
        imgs = imgs.to(self.device)
        feats, attentions = self.dino.get_feat_last_self_attention(imgs)
        feats = feats[:, 1:, :].permute(2,0,1) # N x (patch_size x patch_size) x D
        feats = self.conv(feats).permute(1,2,0)

        # we keep only the output patch attention
        attentions = attentions[:, :, 0, 1:].mean(dim=1).reshape(bs, -1)
        
        if self.threshold is not None:
            val, idx = torch.sort(attentions, descending=False)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval
            idx2 = torch.argsort(idx)
            for index in range(bs):
                th_attn[index] = th_attn[index][idx2[index]]
            th_attn = th_attn.reshape(bs, w_featmap, h_featmap).float()
        points = self.sample_point(th_attn)
        
        feats = feats.reshape(bs, w_featmap, h_featmap, -1).permute(0,3,1,2)
        img_id = torch.Tensor(range(0,points.shape[1])).reshape(points.shape[1], 1)
        
        roi_feats = []
        for i in range(bs):
            feat = feats[i]
            roi_feats_i = []
            for size in range(4, self.max_window_size+1, 2):
                x1, y1 = (points[i,:,0]-size).unsqueeze(1).clamp_(0, 31), (points[i,:,1]-size).unsqueeze(1).clamp_(0, 31)
                x2, y2 = (points[i,:,0]+size).unsqueeze(1).clamp_(0, 31), (points[i,:,1]+size).unsqueeze(1).clamp_(0, 31)
                region = torch.cat([img_id, x1, y1, x2, y2], dim=1).to(self.device) 
                roi_feat = roi_align(feat.unsqueeze(0).repeat_interleave(region.shape[0], 0), region, output_size=(8,8), spatial_scale=0.0625*2)
                roi_feats_i.append(roi_feat.unsqueeze(0))
            roi_feats.append(torch.cat(roi_feats_i, dim=0).unsqueeze(0))
        roi_feats = torch.cat(roi_feats, dim=0).reshape(bs*self.sample_number, -1, 8, 8)
        anchor = F.gumbel_softmax(self.anchor_net(roi_feats), hard=True, eps=1e-20, dim=1)
        gap = torch.matmul(anchor, self.box)

        points = points.reshape(-1, 2).to(self.device)
        x1_new, y1_new = (points[:, 0]-gap).clamp_(0, 31), (points[:, 1]-gap).clamp_(0, 31)
        x2_new, y2_new = (points[:, 0]+gap).clamp_(0, 31), (points[:, 1]+gap).clamp_(0, 31)
        bbox = torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)
        return bbox
    
    def generate_result(self, imgs, mask_imgs, prompts):
        return generate(imgs, mask_imgs, self.pipe, self.generator, prompts, self.device)

    def calculate_clip_loss(self, outputs, target_embeddings):
        # randomly select embeddings
        n_embeddings = np.random.randint(1, len(target_embeddings) + 1)
        target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n_embeddings,))]
        loss = 0.0
        for img in outputs:  # avoid memory limitations
            img_e = self.clip_extractor.get_image_embedding(img.float().unsqueeze(0))
            for target_embedding in target_embeddings:
                loss += self.text_criterion(img_e, target_embedding.unsqueeze(0))

        loss /= len(outputs) * len(target_embeddings)
        return loss
    
    def calculate_clip_dir_loss(self, inputs, outputs, target_embeddings, src_emb):
        # randomly select embeddings
        n_embeddings = np.random.randint(1, min(len(src_emb), len(target_embeddings)) + 1)
        idx = torch.randint(min(len(src_emb), len(target_embeddings)), (n_embeddings,))
        src_embeddings = src_emb[idx]
        target_embeddings = target_embeddings[idx]
        target_dirs = target_embeddings - src_embeddings

        loss = 0.0
        for in_img, out_img in zip(inputs, outputs):  # avoid memory limitations
            in_e = self.clip_extractor.get_image_embedding(in_img.float().unsqueeze(0))
            out_e = self.clip_extractor.get_image_embedding(out_img.float().unsqueeze(0))
            for target_dir in target_dirs:
                loss += 1 - torch.nn.CosineSimilarity()(out_e - in_e, target_dir.unsqueeze(0)).mean()

        loss /= len(outputs) * len(target_dirs)
        return loss
    
    def calculate_structure_loss(self, outputs, inputs):
        loss = 0.0
        for input, output in zip(inputs, outputs):
            with torch.no_grad():
                target_self_sim = self.clip_extractor.get_self_sim(input.float().unsqueeze(0))
            current_self_sim = self.clip_extractor.get_self_sim(output.float().unsqueeze(0))
            loss = loss + torch.nn.MSELoss()(current_self_sim, target_self_sim)

        loss = loss / len(outputs)
        return loss

    def get_loss(self, source_imgs, results, e_prompt, o_prompt):
        text_emb = self.clip_extractor.get_text_embedding(e_prompt, self.device)
        src_emb = self.clip_extractor.get_text_embedding(o_prompt, self.device)
        loss_clip = self.calculate_clip_loss(results, text_emb)        
        loss_dir_clip = self.calculate_clip_dir_loss(source_imgs, results, text_emb, src_emb)
        loss_structure = self.calculate_structure_loss(results, source_imgs)
        
        loss = self.alpha*loss_clip + self.beta*loss_dir_clip + self.gamma*loss_structure
        loss.requires_grad_(True)
       
        return loss, loss_clip, loss_dir_clip, loss_structure

