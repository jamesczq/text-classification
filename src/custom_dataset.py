import numpy as np
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, df, text_col, label_col, max_length):
        """
        Require that 
        df[text_col] is of str type
        df[label_col] is of int type
        """
        self.labels = np.array(df[label_col]) # error w/o np.array(*)!!
        self.texts = [
            tokenizer(
                text, 
                padding='max_length', 
                max_length=max_length, 
                truncation=True, 
                return_tensors="pt") for text in df[text_col]]

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
