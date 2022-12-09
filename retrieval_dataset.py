import os
import torch
import json
from torch.utils.data import Dataset 
from datasets import load_dataset
from PIL import Image

class ITMDataset(Dataset):
    def __init__(self, preprocess, annotations, image_path='sampled_data/images', use_neg_image=False):
        """
        load or save negative images to f'{image_path}/neg_images/'
        """
        self.annotations = annotations
        image_names = os.listdir(image_path)

        self.image_path = image_path
        self.preprocess = preprocess
        self.use_neg_image = use_neg_image
        
        self.neg_img_dir = os.path.join(self.image_path, 'neg_images')
        if use_neg_image and not os.path.exists(self.neg_img_dir):
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
        image = Image.open(self.image_path+"/"+file_name)

        if self.use_neg_image:
            img_idx_str = file_name.split('.')[0]
            neg_img_dir = os.path.join(self.neg_img_dir, img_idx_str)
            if not os.path.exists(neg_img_dir):
                os.makedirs(neg_img_dir)
            
            neg_images = []
            for i in range(len(neg_captions)):
                cap = neg_captions[i]
                neg_img_file = os.path.join(neg_img_dir, f'{i}.jpg')
                if not os.path.exists(neg_img_file):
                    neg_img = self.diffusion_model(cap, guidance_scale=12).images[0]
                    neg_img.save(neg_img_file)
                else:
                    neg_img = Image.open(neg_img_file)
                neg_images.append(neg_img)

            return image, caption, neg_captions, neg_images
        
        return image, caption, neg_captions