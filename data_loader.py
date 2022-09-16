import numpy as np
import pandas as pd

import torch
import transformers

import tqdm

LABELS = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4}

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, df, xcol='text', ycol='category', max_length=128):

        self.labels = [LABELS[label] for label in df[ycol]]
        self.texts = [
            tokenizer(
                text, 
                padding='max_length', 
                max_length=max_length, 
                truncation=True, 
                return_tensors="pt") 
            for text in df[xcol]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y