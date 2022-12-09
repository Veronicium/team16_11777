import os
import torch
import json
from torch.utils.data import Dataset 
from datasets import load_dataset
from PIL import Image

class ITMDataset(Dataset):
    def __init__(self, preprocess, annotations, image_path='sampled_data/images', use_neg_image=False):
        self.annotations = annotations
        image_names = os.listdir(image_path)

        self.image_path = image_path
        self.preprocess = preprocess
        self.use_neg_image = use_neg_image

        if use_neg_image:
            from diffusers import StableDiffusionPipeline
            model_id = "CompVis/stable-diffusion-v1-4"
            print(f"Loading diffusion model: {model_id}")

            # Increasing guidance makes generation follow more closely to the prompt
            self.diffusion_model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation_dict = self.annotations[idx]
        file_name = annotation_dict['file_name']
        caption = annotation_dict['caption']
        neg_captions = annotation_dict['neg_captions']
        # image = self.images[file_name]
        image = Image.open(self.image_path+"/"+file_name)

        if self.use_neg_image:
            # only generate an neg image for the first neg caption to reduce computations
            neg_images = [self.diffusion_model(cap, guidance_scale=12).images[0] for cap in neg_captions]
            print(image, neg_images[0])
            print(caption, neg_captions[0])
            image.save('pos.png')
            neg_images[0].save('neg.png')
            return image, caption, neg_captions, neg_images
        
        return image, caption, neg_captions