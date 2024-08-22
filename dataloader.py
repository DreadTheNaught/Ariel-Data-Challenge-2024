import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from natsort import natsorted

class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_dir, a_transform = None, f_transform = None):
        super().__init__()
        self.f_data = []
        self.a_data = []
        self.a_transform = a_transform
        self.f_transform = f_transform
        self.root_dir = data_dir
        
        self.files = os.listdir(self.root_dir)
        self.files = natsorted(self.files)
        
        self.labels = pd.read_csv(labels_dir)
        
        for planet in self.files:
            
            self.a_data.append(os.path.join(self.root_dir, planet, "AIRS-CH0_signal.parquet" ))
            self.f_data.append(os.path.join(self.root_dir, planet, "FGS1_signal.parquet" ))
        
    def __len__(self):
        if len(self.a_data) == len(self.f_data):
            return len(self.a_data)
    
    def __getitem__(self, index):
        AIRS_data_path = self.a_data[index]
        FGS1_data_path = self.f_data[index]
        
        AIRS_data = pd.read_parquet(AIRS_data_path)
        FGS1_data = pd.read_parquet(FGS1_data_path)
        
        AIRS_data = AIRS_data.values
        FGS1_data = FGS1_data.values
        
        if self.a_transform:
            AIRS_data = self.a_transform(AIRS_data)
        if self.f_transform:
            FGS1_data = self.f_transform(FGS1_data)
        
        return torch.tensor(AIRS_data), torch.tensor(FGS1_data), index