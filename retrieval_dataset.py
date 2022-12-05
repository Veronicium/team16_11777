import os
import torch
import json
from torch.utils.data import Dataset 
from datasets import load_dataset
from PIL import Image

class ITMDataset(Dataset):
    def __init__(self, preprocess, annotations, image_path='sampled_data/images'):
        self.annotations = annotations
        image_names = os.listdir(image_path)

        self.image_path = image_path
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation_dict = self.annotations[idx]
        file_name = annotation_dict['file_name']
        caption = annotation_dict['caption']
        neg_captions = annotation_dict['neg_captions']
        # image = self.images[file_name]
        image = Image.open(self.image_path+"/"+file_name)

        return image, caption, neg_captions