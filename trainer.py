from tqdm import tqdm
import torch
from utils.utility import get_config_data
import os
from datetime import datetime


class train_valid_test():


    def __init__(self, model, train_loader, valid_loader, loss_func) -> None:


        if not os.path.exists('models'):
            os.makedirs('models')


        self.data = get_config_data()
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_func = loss_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model



        self.model.to(self.device)
        
        self.epoch = self.data.get('epoch')

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.data['optimizer']['learning_rate'],
            betas=tuple(self.data['optimizer']['betas']),
            eps=self.data['optimizer'].get('eps', 1e-8),
            weight_decay=self.data['optimizer'].get('weight_decay', 0.0),
            amsgrad=self.data['optimizer'].get('amsgrad', False),
            foreach=self.data['optimizer'].get('foreach', None),
            maximize=self.data['optimizer'].get('maximize', False),
            capturable=self.data['optimizer'].get('capturable', False),
            differentiable=self.data['optimizer'].get('differentiable', False),
            fused=self.data['optimizer'].get('fused', None)
        )


        self.__train_init()

    


    def __train_init(self):

        for i in range(self.epoch):
            running_loss = self.__train_core()

            if i % self.data['chkpt_per_epoch'] == 0:

                chkpt = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
                }

                self.__save_chkpt(chkpt, datetime.now().strftime("%Y%m%d_%H%M%S"))


            print(f"epoch {i + 1}, training - running loss: {running_loss}")


            running_loss = self.__valid()


            print(f"epoch {i + 1}, valid - running loss: {running_loss}")
            



    def __train_core(self):
        loop = tqdm(self.train_loader)
        running_loss = 0.0

        for idx, (inps, label) in enumerate(loop):
            inps = inps.to(self.device)
            label = label.float().to(self.device)

            preds = self.model(inps)
            loss = self.loss_func(preds, label)

            self.optimizer.zero_grad() 
            loss.backward()  
            self.optimizer.step()

            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()

        return running_loss / len(self.train_loader)
    


    def __valid(self):
        self.model.eval()
        loop = tqdm(self.valid_loader)

        running_loss = 0.0

        with torch.no_grad():
            for idx, (x, y) in enumerate(loop):
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)

                loss = self.loss_func(preds, y)


                loop.set_postfix(loss=loss.item())
                running_loss += loss.item()

            return running_loss / len(self.valid_loader)
        

    #Same as Valid
    def __test_init(self):
        pass



    def __save_chkpt(self, state, idx):

        path = f'models/Img_Colrz{idx}.pth.tar'

        torch.save(state, path)

        print('chkpoint saved')