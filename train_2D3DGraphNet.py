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
    "graphnet": "high-capacity", # Possible configuration: "high-capacity", "low-capacity", "GAT"
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

patchnet = PatchNet(hparams["embedding_size"], hparams["pretrained"])
patchnet.to(hparams["device"])

if hparams["graphnet"] == "high-capacity":
    graphnet = GraphNetHighCapacity(num_node_features=6, emb_size=hparams["embedding_size"])
elif hparams["graphnet"] == "low-capacity":
    graphnet = GraphNetLowCapacity(num_node_features=6, emb_size=hparams["embedding_size"])
else:
    graphnet = GraphNetGAT(num_node_features=6, emb_size=hparams["embedding_size"])
graphnet.to(hparams["device"])

parameters = list(patchnet.parameters()) + list(graphnet.parameters())
optimizer = optim.Adam(parameters, lr=hparams["learning_rate"])
criterion = WeightedSoftMarginTripletLoss(hparams["alpha"], hardest=hparams["hardest"]);
criterion = criterion.to(hparams["device"])

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
        dgg = torch.empty(0, hparams["embedding_size"], device=torch.device(hparams["device"]))
        dii = torch.empty(0, hparams["embedding_size"], device=torch.device(hparams["device"]))
        for i, batch in enumerate(loader):
            di = patchnet(batch[1].to(hparams["device"]))
            dg = graphnet(batch[0].to(hparams["device"]), batch[0].batch.to(hparams["device"]))
            loss = criterion(di, dg)
            loss_epoch.append(loss.detach().item())
            dii = torch.cat([dii, di.to(hparams["device"])], dim=0)
            dgg = torch.cat([dgg, dg.to(hparams["device"])], dim=0)
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
total_loss_train = []
total_loss_val = []
total_recall_train = []
total_recall_val = []

start_training = time.time()
for epoch in range(hparams["start_epoch"], hparams["end_epoch"]):
    loss_epoch = []  
    for i, batch in enumerate(train_loader):  
        patchnet.train()
        graphnet.train()
        graphs = batch[0].to(hparams["device"])
        images = batch[1].to(hparams["device"])
        
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

