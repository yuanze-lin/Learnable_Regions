# Text-Driven Image Editing via Learnable Regions <br /> (CVPR 2024)
**:hearts: If you find our project is helpful for your research, please kindly give us a :star2: and cite our paper :bookmark_tabs:   : )**

[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_Text-Driven_Image_Editing_via_Learnable_Regions_CVPR_2024_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16432-b31b1b.svg)](https://arxiv.org/pdf/2311.16432) 
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-0078D4?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/LearnableRegions_page/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mRWzNOlo_RR_zvrnHODgBoPAa59vGUdz#scrollTo=v_4gmzDOzN98)
[![YouTube Video](https://img.shields.io/badge/YouTube%20Video-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=FpMWRXFraK8&feature=youtu.be)

Official implementation of "Text-Driven Image Editing via Learnable Regions" 

[Yuanze Lin](https://yuanze-lin.me/), [Yi-Wen Chen](https://wenz116.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Lu Jiang](http://www.lujiang.info/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)


> Abstract: Language has emerged as a natural interface for image editing. In this paper, we introduce a method for region-based image editing driven by textual prompts, without the need for user-provided masks or sketches. Specifically, our approach leverages an existing pre-trained text-to-image model and introduces a bounding box generator to find the edit regions that are aligned with the textual prompts. We show that this simple approach enables flexible editing that is compatible with current image generation models, and is able to handle complex prompts featuring multiple objects, complex sentences, or long paragraphs. We conduct an extensive user study to compare our method against state-of-the-art methods. Experiments demonstrate the compet- itive performance of our method in manipulating images with high fidelity and realism that align with the language descriptions provided.

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/overview.png)


## Method Overview

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/framework.png)

## News

- [2024.8.16] Release a demo on [Colab](https://colab.research.google.com/drive/1mRWzNOlo_RR_zvrnHODgBoPAa59vGUdz#scrollTo=v_4gmzDOzN98) and have fun playing with it :art:.     
- [2024.8.15] Code has been released.

## Contents


- [Install](#install)
- [Edit Single Image](#edit_single_image)
- [Edit Multiple Images](#edit_multiple_images)
- [Custom Image Editing](#custom_editing)

## Getting Started

### :hammer_and_wrench: Environment Installation <a href="#install" id="install"/>
To establish the environment, just run this code in the shell:
```
git clone https://github.com/yuanze-lin/Learnable_Regions.git
cd Learnable_Regions
conda create -n LearnableRegion python==3.9 -y
source activate LearnableRegion
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda env update --file enviroment.yaml
```
That will create the environment ```LearnableRegion``` we used.

### :tophat: Edit Single Image <a href="#edit_single_image" id="edit_single_image"/>
Run the following command to start editing a single image.

**Since runwayml has removed its impressive inpainting model ('runwayml/stable-diffusion-inpainting'),
if you haven't stored it, please set `--diffusion_model_path 'stabilityai/stable-diffusion-2-inpainting'`.**

```
torchrun --nnodes=1 --nproc_per_node=1 train.py \
	--image_file_path images/1.png \
	--image_caption 'trees' \
	--editing_prompt 'a big tree with many flowers in the center' \
        --diffusion_model_path 'stabilityai/stable-diffusion-2-inpainting' \
	--output_dir output/ \
	--draw_box \
	--lr 5e-3 \
	--max_window_size 15 \
	--per_image_iteration 10 \
	--epochs 1 \
	--num_workers 8 \
	--seed 42 \
	--pin_mem \
	--point_number 9 \
	--batch_size 1 \
	--save_path checkpoints/
```

The editing results will be stored in ```$output_dir```, and the whole editing time of one single image is about 4 minutes with 1 RTX 8000 GPU.  

You can tune `max_window_size`, `per_image_iteration` and `point_number` for adjusting the editing time and performance.

The explanation for the introduced hyper-parameters from our method:

> "**image_caption**": the caption of the input image, we just use class name in our paper.  
>  "**editing_prompt**": the editing prompt for manipulating the input image.  
> "**max_window_size**": max anchor bounding box size.  
> "**per_image_iteration**": training iterations for each image.  
> "**point_number**": number of sampled anchor points.  
> "**draw_box**": whether to draw bounding boxes of results for visualization or not, it will be saved into ```$output_dir/boxes```.

### :space_invader: Edit Multiple Images <a href="#edit_multiple_images" id="edit_multiple_images"/>
Run the following command to start editing multiple images simultaneously.

**If you haven't downloaded the inpaiting model 'runwayml/stable-diffusion-inpainting' before it was closed, please just set `--diffusion_model_path 'stabilityai/stable-diffusion-2-inpainting'`.**

```
torchrun --nnodes=1 --nproc_per_node=2 train.py \
	--image_dir_path images/ \
	--output_dir output/ \
	--json_file images.json \
        --diffusion_model_path 'stabilityai/stable-diffusion-2-inpainting' \
	--draw_box \
	--lr 5e-3 \
	--max_window_size 15 \
	--per_image_iteration 10 \
	--epochs 1 \
	--num_workers 8 \
	--seed 42 \
	--pin_mem \
	--point_number 9 \
	--batch_size 1 \
	--save_path checkpoints/ 
```

### :coffee: How to Edit Custom Images? <a href="#custom_editing" id="custom_editing"/>

**Edit single custom image:** please refer to the command from `Edit Single Image`, and change `image_file_path`, `image_caption`, `editing_prompt` accordingly.

**Edit multiple custom images:** please refer to ```images.json``` to prepare the structure. Each key represents the input image's name, 
the values are class/caption of the input image and editing prompt respectively, and then just run the above command from `Edit Multiple Images`.

## Comparison With Existing Methods

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/comparison.png)

## Results Using Diverse Prompts 

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/results.png)

## Additional Results

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/results2.png)



## Citation

If you find our work useful in your research or applications, please consider citing our paper using the following BibTeX:

```
@inproceedings{lin2024text,
  title={Text-driven image editing via learnable regions},
  author={Lin, Yuanze and Chen, Yi-Wen and Tsai, Yi-Hsuan and Jiang, Lu and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7059--7068},
  year={2024}
}
```
