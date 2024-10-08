from torchvision import transforms
from dataloader import LoadPreprocessed
from torch.utils.data import DataLoader, random_split
from utils.utility import seed_everything, get_config_data
from trainer import train_valid_test
import torch.nn as nn
import torch
from models.BasicCNN import CNN



class MSE_percLoss(nn.Module):

    def __init__(self) -> None:
        super(MSE_percLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, label):
        return self.loss(pred, label) / (label ** 2)
    

def get_transforms():

    a_transform = transforms.Compose([
        transforms.Normalize(mean = [0.0] * data.get('depth'), std = [1.0] * data.get('depth'))
    ])

    f_transform = transforms.Compose([
        transforms.Normalize(mean = [0.0] * data.get('depth'), std = [1.0] * data.get('depth'))
    ])

    return a_transform, f_transform




if __name__ == "__main__":

    data = get_config_data()
    data = data.get('paths')
    seed_everything(data.get('seed_number'))

    a_transform, f_transform = get_transforms()


    dataset = LoadPreprocessed(data_dir=data.get('preprocessed').get('train_data'), 
                               labels_dir=data.get('preprocessed').get('labels'),
                               a_transform=a_transform,
                               f_transform=f_transform)
    

    valid_sz = int(data['valid_sz'] * len(dataset))

    train_ds, valid_ds = random_split(dataset=dataset, lengths=[len(dataset) - valid_sz, valid_sz])

    train_loader = DataLoader(dataset=train_ds, **data['data_loader']['train'])

    valid_loader = DataLoader(dataset=valid_ds, **data['data_loader']['train'])

    
    loss_func = MSE_percLoss()

    del data

    train_valid_test(model = CNN(), train_loader = train_loader, valid_loader = valid_loader, loss_func = loss_func)