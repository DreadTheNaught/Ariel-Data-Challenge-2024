import os
import time
import torch
from dataloader import PreprocessAndLoad


data = PreprocessAndLoad(r'Data/ariel-data-challenge-2024',{'MASK':True,'NLCORR':False,'DARK':True,'TIME_BINNING':True,'FLAT':True},split="train")
start = time.time()
cleaned = r'preprocessed'
os.makedirs(cleaned,exist_ok=True)
for i,j,k in data:
    print(i.shape,j.shape)
    pth = os.path.join(cleaned,str(k))
    os.makedirs(pth,exist_ok=True)
    print(torch.tensor(i).shape)
    torch.save(torch.tensor(i),os.path.join(pth,'AIRS.pth'))
    torch.save(torch.tensor(j),os.path.join(pth,'FGS1.pth'))