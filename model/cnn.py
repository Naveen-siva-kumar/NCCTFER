'''
Aum Sri Sai Ram

Naveen : bnaveensivakumar@gmail.com
'''

import torch
import torch.nn as nn
# import torchvision.models as models
from model.resnet import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class res18feature(nn.Module):
    def __init__(self, args,device=None, pretrained=False, num_classes=7, drop_rate=0.4, out_dim=64):
        super(res18feature, self).__init__()
        
        self.res18  = torch.nn.DataParallel(resnet18(args.num_classes,pretrained)).cuda()
        
        if args.pretrained:
            print("Loading pretrained weights...", args.pretrained) 
            pretrained = torch.load(args.pretrained)
            pretrained_state_dict = pretrained['state_dict']
            model_state_dict = self.res18.state_dict()
            loaded_keys = 0
            total_keys = 0
            for key in pretrained_state_dict:
                if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                    pass
                else:    
                    model_state_dict[key] = pretrained_state_dict[key]
                    total_keys+=1
                    if key in model_state_dict:
                        loaded_keys+=1
                        
            print("Loaded params num:", loaded_keys)
            print("Total params num:", total_keys)
            self.res18.load_state_dict(model_state_dict, strict = False)
        self.drop_rate = drop_rate
        self.out_dim = out_dim
        self.num_classes = num_classes
        

    def forward(self, x,grads=False,num_classes=7):
            if grads:
                f_maps,f = self.res18(x,True)
                return f_maps,f
            new_x = self.res18(x,grads=False)
            
            return new_x
        

       
