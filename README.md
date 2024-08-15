# Text-Driven Image Editing via Learnable Regions <br /> (CVPR 2024)

**[Paper](https://arxiv.org/abs/2311.16432)** | **[Project Page](https://yuanze-lin.me/LearnableRegions_page/)** | **[Youtube Video](https://www.youtube.com/watch?v=FpMWRXFraK8&feature=youtu.be)**

Official implementation of "Text-Driven Image Editing via Learnable Regions" 

[Yuanze Lin](https://yuanze-lin.me/), [Yi-Wen Chen](https://wenz116.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Lu Jiang](http://www.lujiang.info/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)


Abstract: Language has emerged as a natural interface for image editing. In this paper, we introduce a method for region-based image editing driven by textual prompts, without the need for user-provided masks or sketches. Specifically, our approach leverages an existing pre-trained text-to-image model and introduces a bounding box generator to find the edit regions that are aligned with the textual prompts. We show that this simple approach enables flexible editing that is compatible with current image generation models, and is able to handle complex prompts featuring multiple objects, complex sentences, or long paragraphs. We conduct an extensive user study to compare our method against state-of-the-art methods. Experiments demonstrate the compet- itive performance of our method in manipulating images with high fidelity and realism that align with the language descriptions provided. Our project webpage: https://yuanze-lin.me/LearnableRegions_page.

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/overview.png)


## Method Overview

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/framework.png)

## Getting Started

### Installation
To establish the environment, just run this code in the shell:
```
git clone https://github.com/yuanze-lin/Learnable_Regions.git
cd Learnable_Regions
conda create -n LearnableRegion python==3.9
source activate LearnableRegion
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda env update --file enviroment.yaml
```
That will create the environment ```LearnableRegion``` we used.

### Editing Single Image
Run the following command to start training for single image.

Note that you need to fill out ```$huggingface_access_token$``` to successfully run the command.

```
torchrun --nnodes=1 --nproc_per_node=1 train.py \
	--image_file_path images/1.png \
	--image_caption 'trees' \
	--editing_prompt 'a big tree with many flowers in the center' \
	--access_token '$huggingface_access_token$' \
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

The editing results will be stored in ```output_dir```, and the whole training time for editing one single image is about 4 minutes with 1 GPU.

The explanation for hyper-parameters:

> "**image_caption**": the caption of the input image, we just use class name in our paper.  
>  "**editing_prompt**": the editing prompt for manipulating the input image.  
> "**max_window_size**": max anchor bounding box size.  
> "**per_image_iteration**": training iterations for each image.  
> "**point_number**": number of sampled anchor points.  
> "**draw_box**": whether to draw bounding boxes of results for visualization or not, it will be saved into ```output_dir/boxes```.

### Editing Multiple Images 
Run the following command to start training for multiple images.

```
torchrun --nnodes=1 --nproc_per_node=1 train.py \
	--image_dir_path images/ \
	--access_token '$huggingface_access_token$' \
	--output_dir output/ \
	--json_file images.json \
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

### How to Edit Custom Images? 

**Editing single custom image:** please refer to the command from `Editing Single Images`, and change `image_file_path`, `image_caption`, `editing_prompt` accordingly.

**Editing multiple custom images:** please refer to ```images.json``` to prepare the structure. Each key represents the input image's name, 
the values are class/caption of the input image and editing prompt respectively. and then just run the above command from `Editing Multiple Images`.


## Results Using Diverse Prompts 

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/results.png)

## Additional Results

![image](https://github.com/yuanze-lin/LearnableRegions/blob/main/assets/results2.png)



## Citation

```
@article{lin2023text,
  title={Text-Driven Image Editing via Learnable Regions},
  author={Lin, Yuanze and Chen, Yi-Wen and Tsai, Yi-Hsuan and Jiang, Lu and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2311.16432},
  year={2023}
}
```
