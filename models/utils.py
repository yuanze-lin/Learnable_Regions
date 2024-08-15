# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by Henrique Morimitsu
# - Remove unused code
# - Add typing

import io
from typing import Optional, Tuple
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw
import requests
import pdb

def visualize_images(
    images: np.ndarray, 
    title: str = '',
    figsize: Tuple[int, int] = (30, 6)
) -> None:
    batch_size, height, width, c = images.shape
    images = images.swapaxes(0, 1)
    image_grid = images.reshape(height, batch_size*width, c)

    image_grid = np.clip(image_grid, 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(image_grid)
    plt.axis("off")
    plt.title(title)

def read_image_from_url(
    url: str,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> np.ndarray:
    resp = requests.get(url)
    resp.raise_for_status()
    image_bytes = io.BytesIO(resp.content)
    pil_image = Image.open(image_bytes).convert('RGB')
    if height is not None and width is not None:
        pil_image = pil_image.resize((width, height), Image.BICUBIC)
    return np.float32(pil_image) / 255.

def read_image(file_path: str,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> np.ndarray:
    pil_image = Image.open(file_path).convert('RGB')
    if height is not None and width is not None:
        pil_image = pil_image.resize((width, height), Image.BICUBIC)
    return np.float32(pil_image) / 255.

class Bbox(object):
    def __init__(
        self,
        top_left_height_width: str
    ) -> None:
        self.top, self.left, self.height, self.width = [
            int(x) for x in top_left_height_width.split('_')
       ]

def draw_image_with_bbox(
    image: np.ndarray,
    bbox: Bbox
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle((bbox.left, bbox.top), bbox.width, bbox.height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.axis("off")
    plt.title("input")
    plt.show()

def draw_image_with_bbox_new(
    image: np.ndarray,
    bbox: Bbox
) -> Image.Image:
    img = (deepcopy(image)*255).astype(np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox.left, bbox.top, \
            bbox.left+bbox.width, bbox.top+bbox.height
    
    draw.rectangle(((x1, y1), (x2,y2)), outline=(255, 0, 0))
    return img


