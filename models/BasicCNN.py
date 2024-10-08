import torch.nn as nn
import torch
import torch.nn.functional as F


class FGS1_Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        x = x.mean(axis=2)

        return x
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.cnn_block_1 = nn.Sequential(
            nn.Conv1d(283, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        self.cnn_block_2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        self.cnn_block_3 = nn.Sequential(
            nn.Conv1d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        self.fc_block_1 = nn.Sequential(
            nn.Linear(2048 * 23, 2048),
            nn.ReLU()
        )
        
        self.fc_block_2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        
        self.fc_block_3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.fc_block_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fc_block_5 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        self.fc_block_6 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        fgs1 = x["FGS1"]
        airs = x["AIRS"]

        fgs1 = fgs1.mean(axis=3)

        x = torch.concat((fgs1.unsqueeze(2), airs), dim=2)
        x = x.mean(3)
        
        x = x.transpose(1, 2)
        
        # 283, 187
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.cnn_block_3(x)
        
        x = nn.Flatten(0)(x)
        
        x = self.fc_block_1(x)
        x = self.fc_block_2(x)
        x = self.fc_block_3(x)
        x = self.fc_block_4(x)
        x = self.fc_block_5(x)
        x = self.fc_block_6(x)
        
        return x
        
        
