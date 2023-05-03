'''
Aum Sri Sai Ram
#taking consistency on top 3 not bottom three classes

Naveen
'''

# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cnn import * 

from common.utils import accuracy
import os
from algorithm.loss import * 

import copy
from tqdm import tqdm



class noisyfer:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        self.eps = args.eps
        self.warmup_epochs = args.warmup_epochs
        self.alpha = args.alpha 
        self.device =  device#torch.device('cuda:{}'.format(args.gpu))
        self.print_freq = args.print_freq        
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.num_classes  = args.num_classes
        self.max_epochs = args.n_epoch
        self.lr2 =args.lr2
        self.k = args.k

        if  args.model_type=="res":
            #self.model = resModel_50(args)   
            self.model = res18feature(args,self.device)
            #self.ema_model  = resModel(args)
            

        self.model = self.model.to(device)
        
        self.positive_classifier = torch.nn.Linear(512, self.num_classes).to(self.device)
        self.negative_classifier = torch.nn.Linear(512, self.num_classes).to(self.device)
        
        
        self.weighted_CCE =  Sai_weighted_CCE_(num_class=args.num_classes,device = self.device, reduction='mean')
        if args.resume:
            if os.path.isfile(args.resume): #for 3 models
                pretrained = torch.load(args.resume)
                pretrained_state_dict1 = pretrained['model']    
                
                model1_state_dict =  self.model.state_dict()
                loaded_keys = 0
                total_keys = 0
                for key in pretrained_state_dict1: 
                    #print(key)   
                    if  ((key=='module.fcx.weight')|(key=='module.fcx.bias')):
                        print(key)
                        pass
                    else:    
                        model1_state_dict[key] = pretrained_state_dict1[key]
                        total_keys+=1
                        if key in model1_state_dict :
                            loaded_keys+=1
            print("Loaded params num:", loaded_keys)
            
            self.model.load_state_dict(model1_state_dict) 
            linear_layer_dict = pretrained["positive_classifier"]
            pc_state_dict = self.positive_classifier.state_dict()
            for key in linear_layer_dict:
                pc_state_dict[key] = linear_layer_dict[key]
                
            
            self.positive_classifier.load_state_dict(pc_state_dict)
            linear_layer_dict = pretrained["negative_classifier"]
            
            nc_state_dict = self.negative_classifier.state_dict()
            for key in linear_layer_dict:
                nc_state_dict[key] = linear_layer_dict[key]
                
            
            self.negative_classifier.load_state_dict(nc_state_dict)
                
            print('Model loaded from ',args.resume)
    
        p_class_params = self.positive_classifier.parameters()
        n_class_params = self.negative_classifier.parameters()
        
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': p_class_params, 'lr': learning_rate},{'params': n_class_params, 'lr': learning_rate}], lr=self.lr2)
                
                                           
        print('\n Initial learning rate is:')
        for param_group in self.optimizer.param_groups:
            print( "learning rate in optimizer are :", param_group['lr'])                              
        
                
        
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
        self.m1_statedict =  self.model.state_dict()
        self.o_statedict = self.optimizer.state_dict()  
        self.adjust_lr = args.adjust_lr
    
    
    def consistency_loss(self,n_logits_w,n_logits_s):

        pseudo_label = F.softmax(n_logits_w, dim=1)
        n_logits_s_for_consistency_loss = copy.copy(n_logits_s)
        indices_to_keep = pseudo_label <= torch.topk(pseudo_label,k=self.k,largest=False,sorted=False)[0][...,-1,None]
        mask = torch.zeros_like(pseudo_label)
        mask[indices_to_keep] = float(1)
        n_probs_s_for_consistency_loss = F.log_softmax(n_logits_s_for_consistency_loss, dim=1)
        
        pairwise_loss = pseudo_label*n_probs_s_for_consistency_loss*mask
        
        if pairwise_loss.numel():
            return -torch.sum(pairwise_loss,dim=1).mean()
        else:
            return -torch.sum(pairwise_loss,dim=1)
               
               
        
    # Evaluate the Model
    def evaluate(self, test_loader):        #evaluating only from the outputs of positive class classifier.
        print('Evaluating ...')
        self.model.eval()  
        correct1 = 0
        total1 = 0
        correct  = 0
        with torch.no_grad():
            for images,_, labels, _ in test_loader:
                images = (images).to(self.device)
                features = self.model(images)
                logits_w = self.positive_classifier(features)
                outputs1 = F.softmax(logits_w, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()
                _, avg_pred = torch.max(outputs1, 1)
                correct += (avg_pred.cpu() == labels).sum()
                
            acc1 = 100 * float(correct1) / float(total1)
           
           
        return acc1
       
    def save_model(self, epoch, acc, noise,args,time_stamp):  #to save model
        
        torch.save({' epoch':  epoch,
                    'model': self.m1_statedict,
                    "positive_classifier":self.positive_classifier.state_dict(),
                    "negative_classifier":self.negative_classifier.state_dict(),
                    'optimizer':self.o_statedict,},                          
                    'checkpoints/'+args.cp_file+ "/epoch_"+str(epoch)+"_k_"+str(args.k)+'_noise_'+noise+str(time_stamp)+str(args.comment)+"_"+"_"+args.dataset+"_"+"_acc_"+str(acc)[:6]+".pth") 
        print('Models saved at'+'checkpoints/'+args.cp_file+ "/epoch_"+str(epoch)+"_k_"+str(args.k)+'_noise_'+noise+str(args.comment)+"_acc_"+str(acc)[:6]+".pth")
    
    
    # Train the Model
    def train(self, train_loader, epoch,k=1):
        
        print('Training ...')
        self.model.train() 
        if self.adjust_lr:
            if epoch%10==0:
                self.adjust_learning_rate(self.optimizer, epoch)
        
        train_total = 0
        train_correct = 0
        
        if epoch < self.warmup_epochs:
            print('\n Warm up stage using supervision loss based on easy samples')
        elif epoch == self.warmup_epochs:
            print('\n Robust learning stage')
        
        bar = bar = tqdm(range(len(train_loader))) 
        for i, (images_w, images_s, labels, indexes) in enumerate(train_loader):
            images_w = images_w.to(self.device)
            images_s = images_s.to(self.device)
            labels = labels.to(self.device)
            # Forward + Backward + Optimize
            features_w = self.model(images_w)
            features_s = self.model(images_s)
            
            logits_w = self.positive_classifier(features_w)
            logits_s = self.positive_classifier(features_s)
            
            n_logits_w = self.negative_classifier(features_w)
            n_logits_s = self.negative_classifier(features_s)

            if epoch < self.warmup_epochs: 
                              
                loss_p = (self.ce_loss(logits_w,labels) + self.ce_loss(logits_s, labels))/2.0
                
                loss = (loss_p )#+ loss_n)/2.0
               
            else:
                
                loss_p_w, noisy_idx,conf_idx=self.weighted_CCE(logits_w,labels)
                loss_p_s, _,_=self.weighted_CCE(logits_s,labels)
                loss_p = (loss_p_w + loss_p_s)/2.0   
                if noisy_idx.numel():
                    loss_cons_n = self.consistency_loss(n_logits_w[noisy_idx],n_logits_s[noisy_idx])
                
                    loss =  loss_p    +  loss_cons_n 
                else:
                    loss = loss_p
            
            
            prec1 = accuracy(logits_w, labels, topk=(1,))
            train_total += 1
            train_correct += prec1
               
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            bar.set_description(" Epoch:[{epoch:.2f}/{total_epoch:.2f}] Iter:[{iter:.2f}/{total_iter:.2f}]  Loss:{loss:.4f}| Acc:{top1: .4f}".format(
                epoch = epoch+1,
                total_epoch = self.n_epoch,
                iter = i+1,
                total_iter = len(train_loader),
                loss = loss.data.item(),
                top1=prec1.item(),
            ))
            bar.update()
        bar.close()
        return float(train_correct) / float(train_total)
            
    def adjust_learning_rate(self, optimizer, epoch):
        
        for param_group in optimizer.param_groups:
           param_group['lr'] /= 10
           
