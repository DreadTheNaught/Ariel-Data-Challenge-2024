import torch.nn as nn
import torch
import torch.nn.functional as F

class FIR1_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(x):
        x = x.mean(axis = 2)
        
        return x

class INCEPTIONTIME(nn.Module):
    def __init__(self, batch_size=64,
                 nb_filters=187, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        
        super().__init__()

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        
        self.batch_norm = nn.BatchNorm1d(self.nb_filters)
    
    def _inception_module(self, input_tensor, stride=1):
        if self.use_bottleneck:
            input_inception = nn.Conv1d(in_channels=input_tensor.shape[1], 
                                        out_channels=self.bottleneck_size, 
                                        kernel_size=1, 
                                        padding="same", 
                                        bias="false")(input_tensor)
        
        else:
            input_inception = input_tensor
            
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        
        conv_list = []
        
        for i in range(len(kernel_size_s)):
            conv_list.append(nn.ReLU()(
                nn.Conv1d(
                in_channels=int(input_inception.shape[1]),
                out_channels=self.nb_filters,
                kernel_size=kernel_size_s[i],
                stride=stride,
                padding="same",
                bias=False
            )(input_inception)))
            
        max_pool_1 = nn.MaxPool1d(kernel_size=3, padding=1, stride=stride)(input_tensor)
        max_pool_1 = nn.ReLU()(max_pool_1)
        
        conv_6 = nn.Conv1d(int(max_pool_1.shape[1]), self.nb_filters, kernel_size= 1, padding= "same", bias=False)(max_pool_1)
        conv_6 = nn.ReLU()(conv_6)
        
        conv_list.append(conv_6)
        x = torch.concat(conv_list, dim=2)
        print(x.shape)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = nn.ReLU()(x)
        return x
        
    def _shorctcut_layer(self, input_tensor, output_tensor):
        x = nn.Conv1d(in_channels=input_tensor.shape[1],
                      out_channels=self.nb_filters,
                      kernel_size=1, padding='same', bias=False)(input_tensor)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = nn.ReLU(x + output_tensor)
        return x

    def forward(self, x):
        residual = x
        for d in range(self.depth):
            x = self._inception_module(x)
            # print(x.shape)
            if self.use_residual and d % 3 == 2:
                x = self._shorctcut_layer(residual, x)
                residual = x
        
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        return x
