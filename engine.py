from torchvision.utils import save_image
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from torch import autocast
from PIL import Image
from typing import List, Optional, Union
import inspect
import pdb
import numpy as np
import cv2
import torch
import torchvision.transforms as T

def inpaint(pipe, prompts, init_images, mask_images=None, latents=None, strength=0.75, guidance_scale=7.5, generator=None, 
	num_samples=1, n_iter=1):
    all_images = []
    transform = T.PILToTensor()
    for _ in range(n_iter):
        with autocast("cuda"):
            images = pipe(
                prompt=prompts,
                image=init_images,
                mask_image=mask_images,
                generator=generator,
            ).images
    for i in range(len(images)):
        all_images.append(transform(images[i]).unsqueeze(0))
    
    return torch.cat(all_images, dim=0)

def init_diffusion_engine(model_path, device):
    print('Initializing diffusion model: ', model_path)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        revision="fp16", 
        torch_dtype=torch.float16
    ).to(device)

    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(0)
    return pipe, generator

def generate(init_images, mask_images, pipe, generator, prompt=['lion'], device='cuda', strength=0.75, 
    guidance_scale=7.5, num_samples=1, n_iter=1):

    img_size = 512
    transform = T.Resize(img_size)
    init_images, mask_images = transform(init_images), transform(mask_images)
    
    mask_images = mask_images[:,0,:,:].unsqueeze(1)
    results = inpaint(pipe, [prompt]*mask_images.shape[0], init_images, mask_images, strength=strength, guidance_scale=guidance_scale, generator=generator, num_samples=num_samples, n_iter=n_iter)

    return results
