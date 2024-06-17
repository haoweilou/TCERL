import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GIN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
class Attention(nn.Module):
    """Some Information about Attention"""
    def __init__(self,input_dim = 128,hidden_dim=512,dropout=0.5,sum=True):
        super(Attention, self).__init__()
        self.w_omega = nn.parameter.Parameter(torch.normal(mean=0,std=0.1,size=(input_dim,hidden_dim)))
        self.b_omega = nn.parameter.Parameter(torch.normal(mean=0,std=0.1,size=(hidden_dim,)))
        self.u_omega = nn.parameter.Parameter(torch.normal(mean=0,std=0.1,size=(hidden_dim,)))
        self.drop = nn.Dropout(dropout)
        self.sum = sum

    def forward(self, x):
        #The shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = torch.tanh(torch.tensordot(x,self.w_omega,dims=1)+self.b_omega)
        vu = torch.tensordot(v, self.u_omega, dims=1)  # (B,T) shape
        alphas = torch.softmax(vu, dim=1)         # (B,T) shape
        x = x*alphas.unsqueeze(-1)
        x = self.drop(x)
        if self.sum:return torch.sum(x,dim=1)
        else: return x

class TCERL(nn.Module):
    """Some Information about GNNModule"""
    def __init__(self,num_channel=64,num_class=2):
        super(TCERL, self).__init__()
        self.edges,self.weights = None,None
        self.dropout = 0.5
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channel,num_channel,20,stride=2, padding = 1),
            nn.ELU(),
            nn.BatchNorm1d(num_channel),
            nn.Conv1d(num_channel,num_channel,20,stride=2,padding="valid"),
            nn.ELU(),
            nn.BatchNorm1d(num_channel),
            nn.Conv1d(num_channel,num_channel,6,1,"valid"),
            nn.ELU(),
            nn.AvgPool1d(3, stride=2),
            nn.Conv1d(num_channel,num_channel,6,1,"valid"),
            nn.ELU(),
            nn.Dropout(self.dropout)
        )
        hid_dim = 35
        self.attn = nn.Sequential(
            Attention(hid_dim,256,self.dropout),
            nn.ELU(),
            nn.Dropout(self.dropout)
        )

        self.gnn = GIN(hid_dim,256,3,hid_dim,self.dropout)
       
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(self.dropout),
            nn.Linear(hid_dim*num_channel,256),
            nn.ELU(),
            nn.Linear(256,64),
            nn.ELU(),
            nn.Linear(64,16),
            nn.ELU(),
            nn.Linear(16,num_class),
        )
        #Edge construction, change as needed
        complete = torch.tensor([[i,j] for i in range(64) for j in range(64)]).T
        self.edges = complete.to("cuda")

    def forward(self, x):
        #Input size: 20*10*64*400
        x = self.hidden(x)
        #20*64*hid_dim 
        x = self.classifier(x)
        return x

    def hidden(self,x):
        '''
        Input: Batch_size, #Crop slice, #Channels, #Time points
        Example: [20,10,64,400]
        '''
        x_shape = x.shape
        #20*10*64*400 => 200*64*400
        x = torch.reshape(x,(-1,x_shape[2],x_shape[3]))
        #200*64*400
        x  = self.cnn(x)
        #200*64*400 => 20*10*64*400 
        x = torch.reshape(x,(x_shape[0],x_shape[1],x_shape[2],-1))
        #20*64*400 
        x = self.attn(x)
        loader = DataLoader([Data(x,edge_index=self.edges,num_node=x_shape[-2]) for x in x],batch_size=x.shape[0])
        data = next(iter(loader))
        gnn = self.gnn(data.x,data.edge_index)
        #20, num_class
        x = to_dense_batch(gnn,data.batch)[0]
        return x