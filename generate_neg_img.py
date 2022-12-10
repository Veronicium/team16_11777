import os
import torch
import json
from torch.utils.data import Dataset 
from datasets import load_dataset
from PIL import Image
from diffusers import StableDiffusionPipeline


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")
print(f"Finished loading model {model_id}")

data_paths = ['sampled_train_data', 'sampled_val_data']
for data_path in data_paths:
    neg_img_dir = os.path.join(data_path, 'neg_images')

    annotation_file = os.path.join(data_path, 'annotations.json')
    annotations = json.load(open(annotation_file))

    total_num = len(annotations)
    for idx in range(total_num):
        annotation_dict = annotations[idx]
        file_name = annotation_dict['file_name']
        neg_captions = [cap.lower() for cap in annotation_dict['neg_captions']]

        img_idx_str = file_name.split('.')[0]
        neg_img_sub_dir = os.path.join(neg_img_dir, img_idx_str)
        if not os.path.exists(neg_img_sub_dir):
            os.makedirs(neg_img_sub_dir)
        
        for i in range(len(neg_captions)):
            cap = neg_captions[i]
            neg_img_file = os.path.join(neg_img_sub_dir, f'{i}.jpg')
            if not os.path.exists(neg_img_file):
                neg_img = pipe(cap, guidance_scale=12).images[0]
                neg_img.save(neg_img_file)
