# TCERL
This repository is the official PyTorch implementation of the GNC paper:Transferrable Subject-Independent Feature Representation for Discriminating EEG-Based Brain Signals, authored by Haowei Lou, Zesheng Ye, and Lina Yao.

**Abstract**: Subject-independent electroencephalography (EEG) recognition remains challenging due to inherent variability of brain anatomy across different subjects. Such variability is further complicated by the volume conductor effect (VCE) that introduces channel-interference noise, exacerbating subject-specific biases in the recorded EEG signals. Existing studies, often relying large datasets and entangled spatial-temporal features, struggle to overcome this bias, particularly in scenarios with limited EEG data. To this end, we propose a Temporal-Connective EEG Representation Learning (TCERL) framework that disentangles temporal and spatial feature learning. TCERL first employs an one-dimensional convolutional network to extract channel-specific temporal features, mitigating channel-interference noise caused by VCE. Building upon these temporal features, TCERL then leverages Graph Neural Networks to extract subject-invariant topological features from a functional brain network, constructed using the channel-specific features as nodes and functional connectivity as the adjacency matrix. This approach allows TCERL to capture robust representations of brain activity patterns that generalize well across different subjects. Our empirical experiment demonstrates that TCERL outperforms state-of-the-art across a range of training subjects on four public benchmarks and is less sensitive to subject variability. The performance gain is highlighted when limited subjects are available, suggesting the robustness and transferability of the proposed method.

**Method**: We first crop raw EEG signal into a sequence of the temporal slice using the slide window technique; adopt 1D-CNN to extract temporal features and the self-attentive module to search the most discriminative temporal slice; then we combine nodes in temporal embedding with functional connectivity to generate the graph representation of bain network; select three layer of graph neural network (GNN) to extract topological features and lastly the extracted topological features are classified to different motion intention using a fully connected network with softmax activation function.

![Architecature diagram](/fig/overview.png)

## Dataset
* For EEGMMIDB, download the dataset from [Physionet](https://physionet.org/content/eegmmidb/1.0.0/) and unzip the download file into `dataset` directory or modify the path in `getEEGMMIDB` function in `dataset.py`
* For custom dataset, process the dataset containing EEG signals in the shape [M,N,K] and corresponding label. Then, follow the EEGMMIDB example to prepare the PyTorch dataset.

## Environment
* Python - 3.11.9
* Pytorch - 2.3.1

`pip install -r requirements.txt`

## Training
* Critical Edge formation

```
x = EEGSIGNAL
y = label
t = selection ratio
edge = critical_edge(x,y,t,self_loop=self_loop)
```

* Start Training
`python run.py`

## Citation
