import torch.nn as nn
import torch.nn.functional as F
import os
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import accuracy_score,classification_report, roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
class Train():
    """Some Information about MAML"""
    def __init__(self, model:nn.Module,train_dataset,test_dataset,model_name="TCERL"):
        self.device = "cuda"
        self.model_name = model_name
        self.model = model
        self.lr = 0.001

        self.loss_func = nn.CrossEntropyLoss()
        self.epochs = 120
        self.batch_size = 500

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    
    def torch_loader(self,dataset:Data.Dataset):
        return Data.DataLoader(dataset,self.batch_size,shuffle=True)

    def forward(self):
        model = deepcopy(self.model)
        out_epoch = 120
        #train task
        root = f"./"
        if not os.path.exists(root): os.mkdir(root)
        log = {"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[],"test_auc":[],"test_f1":[]}
        for epoch in tqdm(range(out_epoch)):
            train_loss, train_acc = self.train(model,self.torch_loader(self.train_dataset))
            test_loss, test_acc, test_auc,test_f1 = self.test(model,self.torch_loader(self.test_dataset))
            log["train_loss"].append(train_loss)
            log["train_acc"].append(train_acc)
            log["test_loss"].append(test_loss)
            log["test_acc"].append(test_acc)
            log["test_auc"].append(test_auc)
            log["test_f1"].append(test_f1)
            train_log = pd.DataFrame(log)
            train_log.to_csv(f"{root}{self.model_name}.csv")
        #evaluation
        return 0
    
     
    def train(self,model,trainloader,num_class=2):
        train_y = []
        predict_y = []
        train_loss = 0
        model.train()

        opt = optim.Adam(model.parameters(), lr=self.lr)
        
        for data, label in trainloader:
            label_onehot = F.one_hot(label.long(),num_classes=num_class).float().to(self.device)
            output = model(data.to(self.device).float())
            loss = self.loss_func(output,label_onehot)  # cross entropy loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            output = np.argmax(output.cpu().detach().numpy(),axis=1)
            predict_y.append(output)
            train_y.append(label.cpu().detach().numpy())
        
        train_y = np.concatenate(train_y)
        predict_y = np.concatenate(predict_y)
        train_loss = train_loss/len(trainloader)
        train_acc = accuracy_score(train_y,predict_y)
        return train_loss, train_acc

    def test(self,model,testloader,num_class=2,report=False):
        test_y = []
        predict_y = []
        test_loss = 0
        model.eval()
        for data, label in testloader:
            #print(data.shape)
            label_onehot = F.one_hot(label.long(),num_classes=num_class).float().to(self.device)
            output = model(data.to(self.device).float())
            loss = self.loss_func(output,label_onehot)  # cross entropy loss
            test_loss += loss.item()
            output = np.argmax(output.cpu().detach().numpy(),axis=1)
            predict_y.append(output)
            test_y.append(label.cpu().detach().numpy())
        test_y = np.concatenate(test_y)
        predict_y = np.concatenate(predict_y)
        if report: return classification_report(test_y, predict_y)
        test_loss = test_loss/len(testloader)
        test_acc = accuracy_score(test_y,predict_y)
        test_auc = roc_auc_score(test_y,predict_y)
        test_f1 = f1_score(test_y,predict_y)
        return test_loss, test_acc, test_auc,test_f1
    
