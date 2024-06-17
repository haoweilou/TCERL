import torch
from model import TCERL
from train import Train
from dataset import EEGMMIDB
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TCERL().to(device)
subjects = [i for i in range(1,110) if i not in [88,89,92,100]]
train_dataset, test_dateset = EEGMMIDB(subjects[:90]), EEGMMIDB(subjects[90:])
model.edges = train_dataset.edges.to(device)
edges = model.edges.tolist()
with open("edges","w") as f: 
    f.write(str(edges))
train = Train(model,train_dataset,test_dateset)
train.forward()
