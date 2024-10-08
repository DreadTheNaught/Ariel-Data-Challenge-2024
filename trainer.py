from tqdm import tqdm
import torch
from utils.utility import get_config_data
import os
from datetime import datetime
import wandb


class train_valid_test():


    def __init__(self, model, train_loader, valid_loader, loss_func) -> None:


        if not os.path.exists('checkpts'):
            os.makedirs('checkpts')


        self.data = get_config_data()
        self.data = self.data.get('paths')
        
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
            betas=self.data['optimizer']['betas'],
            eps=self.data['optimizer'].get('eps', 1e-8),
            weight_decay=self.data['optimizer'].get('weight_decay', 0.0),
            amsgrad=self.data['optimizer'].get('amsgrad', False),
            foreach=self.data['optimizer'].get('foreach', None),
            maximize=self.data['optimizer'].get('maximize', False),
            capturable=self.data['optimizer'].get('capturable', False),
            differentiable=self.data['optimizer'].get('differentiable', False),
            fused=self.data['optimizer'].get('fused', None)
        )

        
        wandb.login(key=self.data.get('wandb_login_key'))
        wandb.init(project=self.data.get('wandb_project'), name=self.data.get('wandb_project_name'))

        wandb.config.update({
        "learning_rate": self.data['optimizer']['learning_rate'],
        "epochs": self.epoch,
        })


        
        wandb.watch(self.model, log='all', log_freq=50)

        self.grad_norms = {name: [] for name, prm in self.model.named_parameters()}


        self.__train_init()


        wandb.finish()

    


    def __train_init(self):

        for i in range(self.epoch):
            running_loss = self.__train_core()

            if i % self.data['chkpt_per_epoch'] == 0:

                chkpt = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
                }

                self.__save_chkpt(chkpt, datetime.now().strftime("%Y%m%d_%H%M%S"))


            print(f"epoch: {i + 1}, training-loss: {running_loss}")

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    self.grad_norms[name].append(grad_norm)

                    wandb.log({f"grad_norms/{name}": grad_norm}, step=i + 1)


            valid_loss = self.__valid()


            wandb.log({
            "train_loss": running_loss,
            "valid_loss": valid_loss,
            }, step=i + 1)


            print(f"epoch: {i + 1}, valid-loss: {valid_loss}")
            



    def __train_core(self):
        loop = tqdm(self.train_loader)
        running_loss = 0.0

        for idx, data in enumerate(loop):
            inps = data["signals"]
            label = data["labels"]
            
            loss = self.__run_batch(inps, label)

            loop.set_postfix(loss=loss.item())
            running_loss += loss.detach().item()

        return running_loss / len(self.train_loader)
    

    def __run_batch(self, inps, label):
        
        inps["FGS1"] = inps["FGS1"].to(self.device)
        inps["AIRS"] = inps["AIRS"].to(self.device)
        label = label.float().to(self.device)
        print(inps["FGS1"].shape, inps["AIRS"].shape)
        preds = self.model(inps)
        loss = self.loss_func(preds, label)

        self.optimizer.zero_grad() 
        loss.backward()  
        self.optimizer.step()

        return loss.item()


    


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
                running_loss += loss.detach.item()

            return running_loss / len(self.valid_loader)
        

    #Same as Valid
    def __test_init(self):
        pass



    def __save_chkpt(self, state, idx):

        path = f'checkpts/Img_Colrz{idx}.pth.tar'

        torch.save(state, path)

        print('chkpoint saved')