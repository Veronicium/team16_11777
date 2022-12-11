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
        self.neg_img_dir = os.path.join(image_path.split('images')[0], 'neg_images')
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation_dict = self.annotations[idx]
        file_name = annotation_dict['file_name']
        caption = annotation_dict['caption'].lower()
        neg_captions = [cap.lower() for cap in annotation_dict['neg_captions']]
        image = Image.open(self.image_path+"/"+file_name)

        if self.use_neg_image:
            neg_images = []
            for i in range(len(neg_captions)):
                neg_img_file = os.path.join(os.path.join(self.neg_img_dir, file_name.split('.jpg')[0]), f'{i}.jpg')  
                temp = Image.open(neg_img_file)
                neg_img = temp.copy()
                neg_images.append(neg_img)
                temp.close()
            return image, caption, neg_captions, neg_images
        
        return image, caption, neg_captions