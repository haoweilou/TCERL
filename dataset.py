import logging
import numpy as np
from scipy import stats
from tqdm import tqdm
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
import torch
import pandas as pd
def getEEGMMIDB(subjects,need_tqdm = True,run = [4,8,12]):
    logger = logging.getLogger('mne')
    logger.disabled = True
    all_data = []
    all_label = []
    if(need_tqdm): 
        pbar = tqdm(subjects)
    else: 
        pbar = subjects
    
    for subject in pbar:
        raw = [f"./dataset/files/S{str(subject).zfill(3)}/S{str(subject).zfill(3)}R{str(i).zfill(2)}.edf" for i in run]
        raw = concatenate_raws([read_raw_edf(f,preload=True) for f in raw])
        
        channel_names = raw.ch_names
        channel_names = [i.replace(".","") for i in channel_names]
        
        #set sensor location
        eegbci.standardize(raw)
        montage = make_standard_montage('standard_1020')
        raw.set_montage(montage)

        raw.rename_channels(lambda x: x.strip('.'))

        # Apply band-pass filter
        #load motor imagery related event
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
        labels = events[:,-1:].copy()
        labels -= 2
        epochs = mne.Epochs(raw, events, [2,3], 0.1,3.2, proj=True, picks=picks, baseline=None, preload=True)
        data = epochs.get_data(copy=False)
        all_data.append(data)
        all_label.append(labels)
    all_data = np.concatenate(all_data)
    
    all_data = stats.zscore(all_data)
    all_label = np.concatenate(all_label)
    all_label = all_label[all_label >= 0]
    
    return all_data,all_label

def critical_edge(data,label,t=0.5,self_loop=True):
    # Task-relevant edge seletion algorithm
    # data: [batch,#channel,#timepoints]
    # label: [batch,1]
    label = label.squeeze(1) if len(label.shape) == 2 else label
    num_label = int(np.max(label)+1)
    num_channel = data.shape[1]
    E = []
    num_edges = num_channel*num_channel
    for label_ in range(num_label):
        subset = data[label==label_]
        xc = np.array([np.corrcoef(s) for s in subset])
        xc = np.abs(xc)
        xc = np.nanmean(xc,axis=0)
        if self_loop: triu  = [[i,j, xc[i][j]] for i in range(num_channel) for j in range(num_channel)]
        else: triu  = [[i,j, xc[i][j]] for i in range(num_channel) for j in range(num_channel) if i != j]
        xc_star = sorted(triu,key=lambda x:x[2],reverse=True)
        Ec = [(x[0],x[1],x[2]) for x in xc_star[:int(num_edges*t)]]
        E = list(set(E)|set(Ec))

    E = pd.DataFrame(E)
    E = E.groupby([0,1]).mean().reset_index()
    E = E.to_numpy()
    E = np.array([(int(i[0]),int(i[1]),i[2]) for i in E])
    E = np.array([E[:,0],E[:,1],E[:,2]])
    edge,weight =  E[:2],E[2]
    return edge.astype(np.int32),weight

def crop_data(data:np.ndarray,segment_length=16, moving=10):
    """
    slide window crop
    """
    data_length = data.shape[1]
    output = []
    start = 0
    end = start + segment_length
    while end < data_length:
        output.append(data[:,start:end])
        start += moving
        end += moving
    return np.array(output)

class EEGMMIDB(torch.utils.data.Dataset):
    """EEGMMIDB dataset"""
    def __init__(self,subjects=[i for i in range(1,110) if i not in [88,89,92,100]],t=0.3,crop=True,need_tqdm=True):
        #t : top t percent most-critical edges
        super(EEGMMIDB, self).__init__()
        self.edges,self.weight = None, None
        self.need_tqdm = need_tqdm
        
        self.data,self.label = self.processData(subjects,crop=crop,t=t)

        if not crop:self.data = np.expand_dims(self.data,1)
        self.label = np.squeeze(self.label)
        self.shape = self.data.shape

        
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.shape[0]

    def processData(self,subjects,crop=True,t=0.3,self_loop=True):
        data,label = getEEGMMIDB(subjects,self.need_tqdm)
        
        self.edges,self.weight = critical_edge(data,label,t,self_loop=self_loop)
        self.edges = torch.tensor(self.edges,dtype=torch.int64)
        if crop:data = np.array([crop_data(x,segment_length = 400,moving=10) for x in data]).astype(np.float32)
        return data,label