from PIL import Image
from argparse import ArgumentParser
import os
import numpy as np 
import clip
import torch
import torch.distributed as dist
import pdb
import shutil
import torch.nn as nn
import argparse
import utils.vision_transformer as vits
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def l2_loss(output, target):
    loss = torch.pow(torch.abs(output-target),2).mean(dim=1)
    return loss

def init_clip_model(device):
    model, preprocess = clip.load('ViT-B/16', device)
    return model, preprocess

def init_dino_model(device):
    patch_size=16
    dino = vits.__dict__["vit_base"](patch_size=patch_size, num_classes=0).to(device)
    dino.eval()
    url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    dino.load_state_dict(state_dict, strict=True)
    
    return dino

def extract_features(model, dino_model, preprocess, input_img_path, edit_img_dir, input_text, device):
    input_text = clip.tokenize(input_text).to(device)
    img_gather, img_paths = [], []
    img_list = os.listdir(edit_img_dir)
    for img in img_list:
        img_path = os.path.join(edit_img_dir, img)
        try:
            edit_img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except:
            pass
        img_gather.append(edit_img)
        img_paths.append(img_path)

    input_img = preprocess(Image.open(input_img_path)).unsqueeze(0).to(device)
    edit_imgs = torch.cat(img_gather, dim=0)

    with torch.no_grad():
        text_features = model.encode_text(input_text)
        image_features = model.encode_image(edit_imgs)
        input_img_feature = model.encode_image(input_img)
        
        dino_image_features = dino_model.get_feat_last_self_attention(edit_imgs)[0][:,0,:]
        dino_input_img_feature = dino_model.get_feat_last_self_attention(input_img)[0][:,0,:]
        

    input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
    dino_input_img_feature /= dino_input_img_feature.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return input_img_feature, image_features, text_features, img_paths, \
                edit_imgs, input_img, dino_image_features, dino_input_img_feature 

def compute_similarity(input_img_feature, image_features, dino_input_img_feature, dino_image_features, text_features, cosine_sim=True): 
    if cosine_sim == True:
        iti_similarity = torch.nn.functional.cosine_similarity(image_features, input_img_feature).softmax(dim=0)
        tti_similarity = torch.nn.functional.cosine_similarity(image_features, text_features).softmax(dim=0)
        dino_iti_similarity = torch.nn.functional.cosine_similarity(dino_image_features, dino_input_img_feature).softmax(dim=0) 
    else:
        iti_similarity = (100.0 * image_features @ input_img_feature.T).softmax(dim=0)[:, 0]
        tti_similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)[:, 0]
        dino_iti_similarity = (100.0 * dino_image_features @ dino_input_img_feature.T).softmax(dim=0)[:, 0]

    return iti_similarity, dino_iti_similarity, tti_similarity


def get_final_img(args, input_text, input_img_path, edit_img_path, topk_tti=3):
    alpha, beta = args.test_alpha, args.test_beta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = init_clip_model(device)
    dino_model = init_dino_model(device)
    img_save, dic = [], {}
    
    input_img_feature, image_features, text_features, img_paths, imgs, \
            input, dino_image_features, dino_input_img_feature = \
                extract_features(model, dino_model, preprocess, input_img_path, edit_img_path, input_text, device)
                
    iti_similarity, dino_iti_similarity, tti_similarity = compute_similarity(input_img_feature, \
            image_features, dino_input_img_feature, dino_image_features, text_features)
    
    score_tti = alpha * tti_similarity
    indices_tti = score_tti.topk(len(score_tti))[1]
    
    score = alpha * tti_similarity  + beta * iti_similarity
    indices = score.topk(len(score))[1]

    for index in indices:
        # restrict text-to-image similarity
        if index in indices_tti[:topk_tti]:
            img_save.append(img_paths[index])
            break

    rank = dist.get_rank()
    for i in range(len(img_save)):
        input_path = img_save[i]
        img_name =  'final_output.png'
        output_img_path = os.path.join(edit_img_path,img_name)
        if rank == 0:
            shutil.copyfile(input_path, output_img_path)
