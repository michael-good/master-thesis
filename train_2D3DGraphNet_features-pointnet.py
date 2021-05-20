import os
import json
import torch
import time
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict
from torch_geometric.data import DataLoader

from graphnet.graphdataset import GraphDataset
from graphnet.models import *
from graphnet.losses import *


hparams = {
    "root_train": "./data/train",
    "root_val": "./data/train/validation",
    "dest": "./logs",
    "device": "cuda",
    "start_epoch": 0,
    "end_epoch": 80,
    "batch_size": 128,
    "num_workers": 8,
    "num_files": 86,
    "undirected": False,
    "embedding_size": 64,
    "num_samples": 50000,
    "pretrained": True,
    "hardest": False,
    "learning_rate": 0.00006,
    "alpha": 5.0
}

start_time = time.time()

train_dataset = GraphDataset(hparams["root_train"], num_files=hparams["num_files"], k=9, undirected=hparams["undirected"])
train_loader = DataLoader(
    train_dataset,
    batch_size=hparams["batch_size"],
    num_workers=hparams["num_workers"],
    pin_memory=True,
    shuffle=True,
)

val_dataset = GraphDataset(hparams["root_val"], num_files=1, k=9,undirected=hparams["undirected"])
val_loader = DataLoader(
    val_dataset,
    batch_size=hparams["batch_size"],
    num_workers=hparams["num_workers"],
    pin_memory=True,
    shuffle=False,
)

print("> Total dataset loading time: {:.2f} seconds".format(time.time() - start_time))
print("> Length train: {} | Length val: {}".format(len(train_dataset), len(val_dataset)))

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
       

    def forward(self, x):
        # input.shape == (bs,n,3)
        bs = x.shape[0]
        xb = self.relu(self.bn1(self.conv1(x)))
        xb = self.relu(self.bn2(self.conv2(xb)))
        xb = self.relu(self.bn3(self.conv3(xb)))
        #xb = torch.max(xb, 2, keepdim=True)[0]
        #flat = xb.view(-1, 1024)
        pool = nn.MaxPool1d(1024)(xb)
        flat = nn.Flatten(1)(pool)
        xb = self.relu(self.bn4(self.fc1(flat)))
        xb = self.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.to(xb.device)
        # add identity to the output
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix
    
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)
        
        # Changed input size of conv2 layer so that it accepts concatenation of RGB data
        self.conv2 = nn.Conv1d(67,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.relu = nn.ReLU()
       
    def forward(self, coords, colors):
        matrix3x3 = self.input_transform(coords)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(coords,1,2), matrix3x3).transpose(1,2)
        xb = self.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        # Concatenate color data to the output of transformations of size (batch_size, 64, 1024)
        # Shape after concatenation is (batch_size, 67, 1024)
        xb = torch.cat([xb, colors], dim=1)

        xb = self.relu(self.bn2(self.conv2(xb)))
        feature = self.bn3(self.conv3(xb))
        xb = torch.max(feature, 2, keepdim=True)[0]
        output = xb.view(-1, 1024)

        return output, matrix3x3, matrix64x64, feature
    
class PointNet(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_size)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()

    def forward(self, coords, colors):
        xb, matrix3x3, matrix64x64, feature = self.transform(coords, colors)
        xb = self.relu(self.bn1(self.fc1(xb)))
        xb = self.relu(self.bn2(self.fc2(xb)))
        xb = self.fc3(xb)
        output = nn.functional.normalize(xb, p=2, dim=1)
        return output, matrix3x3, matrix64x64, feature.permute(0, 2, 1)

patchnet = PatchNet(hparams["embedding_size"], hparams["pretrained"])
patchnet.to(hparams["device"])
    
graphnet = GraphNetPointNetFeatures(num_node_features=1024, emb_size=hparams["embedding_size"])
graphnet.to(hparams["device"])

# Load a pretrained PointNet model from 2D-3D MatchNet
pointnet = PointNet(hparams["embedding_size"])
fname = "./logs/pointnet_best-recall.pth"
checkpoint_pointnet = torch.load(fname)
pointnet.load_state_dict(checkpoint_pointnet['pointnet'])
pointnet.to(hparams["device"])
pointnet.eval()

parameters = list(patchnet.parameters()) + list(graphnet.parameters())
optimizer = optim.Adam(parameters, lr=hparams["learning_rate"])
criterion = WightedSoftMarginTripletLoss(hparams["alpha"], hardest=hparams["hardest"]);
criterion.to(hparams["device"])

def separate_coords_colors(points):
    coords = torch.empty(0, 3, 1024)
    colors = torch.empty(0, 3, 1024)
    for i in range(points.shape[0]):
        coord = points[i][:3, :]
        color = points[i][3:, :]
        coords = torch.cat([coords, coord.unsqueeze(0)], dim=0)
        colors = torch.cat([colors, color.unsqueeze(0)], dim=0)
    return coords, colors

def recall_epoch(loader, k=6):
    '''
    Given a DataLoader object and an integer k, this function computes recall@k on the dataset provided.
    Parameters:
        - loader: DataLoader object
        - k: Integer that determines how many top samples to take into account in the recall calculation.
    '''
    patchnet.eval()
    graphnet.eval()
    loss_epoch = []
    with torch.no_grad():
        dgg = torch.empty(0, hparams["embedding_size"])
        dii = torch.empty(0, hparams["embedding_size"])
        for i, batch in enumerate(loader):
            graphs = batch[0]
            images = batch[1].to(hparams["device"])    
            
            clouds = torch.empty(0, 1024, 6)
            j = 0
            while j < graphs.x.shape[0]:
                cloud = graphs.x[j:j+1024, :]
                clouds = torch.cat([clouds, cloud.unsqueeze(0)], dim=0)
                j += 1024

            coords, colors = separate_coords_colors(clouds.permute(0, 2, 1))
            features = pointnet(coords.to(hparams["device"]), colors.to(hparams["device"]))[3] 
            features = features.cpu().numpy()
            features = torch.from_numpy(np.concatenate(features))
            graphs.x = features
            graphs = graphs.to(hparams["device"])
            
            di = patchnet(images)
            dg = graphnet(graphs, graphs.batch)
            
            loss = criterion(di, dg)
            loss_epoch.append(loss.item())
            
            dii = torch.cat([dii, di], dim=0)
            dgg = torch.cat([dgg, dg], dim=0)
        xx = torch.sum(torch.pow(dii, 2), 1).view(-1, 1)
        yy = torch.sum(torch.pow(dgg, 2), 1).view(1, -1)
        distances = xx + yy - 2.0 * torch.mm(dii, torch.t(dgg))
        tp, fn = 0, 0
        values, indices = torch.topk(-distances, k)
        for i, row in enumerate(indices):
            if i in row:
                tp += 1
            else:
                fn += 1
        recall = tp / (tp + fn)
        loss = sum(loss_epoch)/len(loss_epoch)
    return loss, recall

best_recall = 0
total_loss = []
total_loss_val = []
total_recall_train = []
total_recall_val = []

start_training = time.time()
for epoch in range(hparams["start_epoch"], hparams["end_epoch"]):
    loss_epoch = []  
    for i, batch in enumerate(train_loader):  
        patchnet.train()
        graphnet.train()
        graphs = batch[0]
        images = batch[1].to(hparams["device"]) 
        
        clouds = torch.empty(0, 1024, 6)
        j = 0
        while j < graphs.x.shape[0]:
            cloud = graphs.x[j:j+1024, :]
            clouds = torch.cat([clouds, cloud.unsqueeze(0)], dim=0)
            j += 1024
        
        coords, colors = separate_coords_colors(clouds.permute(0, 2, 1))
        with torch.no_grad():
            features = pointnet(coords.to(hparams["device"]), colors.to(hparams["device"]))[3] 
            features = features.detach().cpu().numpy()
            features = torch.from_numpy(np.concatenate(features))
        
        graphs.x = features
        graphs = graphs.to(hparams["device"])
        
        dg = graphnet(graphs, graphs.batch)
        di = patchnet(images)

        loss = criterion(di, dg)
        loss_epoch.append(loss.detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        now = datetime.datetime.now()
        log = "{} | Epoch [{:04d}/{:04d}] | Batch [{:04d}/{:04d}] | loss: {:.4f}"
        log = log.format(now.strftime("%c"), epoch, hparams["end_epoch"]-1, i, len(train_loader)-1, loss_epoch[-1])
        print(log, flush=True)
    
    mean_loss_epoch = sum(loss_epoch)/len(loss_epoch)
    total_loss_train.append(mean_loss_epoch)
    
    # Recall computation on train split by sampling hparams[num_samples] random samples from train_dataset
    print(" > Computing recall of epoch [{:04d}/{:04d}] on train split...".format(epoch, hparams["end_epoch"]-1), flush=True)
    rand_indx = torch.randint(0, len(train_dataset), (hparams["num_samples"],))
    shuffled_dataset = torch.utils.data.Subset(train_dataset, rand_indx.tolist())
    samples_train_loader = DataLoader(
        shuffled_dataset, 
        batch_size=hparams["batch_size"], 
        num_workers=hparams["num_workers"], 
        pin_memory=True,
        shuffle=False
    )
    _, recall_train = recall_epoch(samples_train_loader, k=6)
    total_recall_train.append(recall_train)
    
    # Recall computation on validation split
    print(" > Computing recall of epoch [{:04d}/{:04d}] on validation split...".format(epoch, hparams["end_epoch"]-1), flush=True)
    loss_val, recall_val = recall_epoch(val_loader, k=6)
    total_loss_val.append(loss_val)
    total_recall_val.append(recall_val)
    
    # Save current model state if maximum recall
    if recall_val > best_recall:
        fname = os.path.join(hparams["dest"], "model_best-recall.pth")
        print(" > Saving model to {}...".format(fname))
        model = {"graphnet": graphnet.state_dict(), "patchnet": patchnet.state_dict()}
        torch.save(model, fname)
        best_recall = recall_val
    
    # Summary after each epoch
    now = datetime.datetime.now()  
    log = " > {} | Epoch [{:04d}/{:04d}] | Loss: {:.4f} | Recall train: {:.4f} | Recall val: {:.4f}"
    log = log.format(now.strftime("%c"),  epoch, hparams["end_epoch"]-1, mean_loss_epoch, recall_train, recall_val)
    print(log, flush=True)
    print("--------------------------------------------------------------------------")
    
    # Save train log
    fname = os.path.join(hparams["dest"], "train.log")
    with open(fname, 'a') as f:
        f.write(log + "\n")

# Save model into memory once training process is completed
fname = os.path.join(hparams["dest"], "{}_model.pth".format(datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")))
print(" > Saving model to {}...".format(fname))
torch.save({
            'epoch': epoch,
            'graphnet': graphnet.state_dict(),
            'patchnet': patchnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss_train,
            'recall_train': total_recall_train,
            'recall_val': total_recall_val,
            'best_recall': best_recall
            }, fname)

