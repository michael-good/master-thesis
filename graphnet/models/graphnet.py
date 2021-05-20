import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_max_pool

class GraphNetHighCapacity(nn.Module):
    def __init__(self, num_node_features=6, emb_size=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.bn1 = BatchNorm(32)
        self.conv2 = GCNConv(32, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GCNConv(64, 128)
        self.bn3 = BatchNorm(128)
        self.conv4 = GCNConv(128, 1024)
        self.bn4 = BatchNorm(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn5 = BatchNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = BatchNorm(256)
        self.fc3 = nn.Linear(256, emb_size)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index))) 
        x = F.relu(self.bn4(self.conv4(x, edge_index)))  

        x = global_max_pool(x, batch)
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        
        output = F.normalize(x, p=2, dim=1)
        return output

class GraphNetLowCapacity(nn.Module):
    def __init__(self, num_node_features=6, emb_size=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.bn1 = BatchNorm(32)
        self.conv2 = GCNConv(32, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GCNConv(64, 128)
        self.bn3 = BatchNorm(128)
        
        self.drop_layer1 = nn.Dropout(p=0.5)
        self.drop_layer2 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(128, 128)
        self.bn4 = BatchNorm(128)
        self.fc2 = nn.Linear(128, emb_size)


    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))   

        x_flat = global_max_pool(x, batch)
        
        x_flat = self.drop_layer1(F.relu(self.bn4(self.fc1(x_flat))))
        x_flat = self.drop_layer2(self.fc2(x_flat))
        
        output = F.normalize(x_flat, p=2, dim=1)
        return output

class GraphNetGAT(nn.Module):
    def __init__(self, num_node_features=6, emb_size=64):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 8, concat=True, heads=4)
        self.bn1 = BatchNorm(32)
        self.conv2 = GATConv(32, 16, concat=True, heads=4)
        self.bn2 = BatchNorm(64)
        self.conv3 = GATConv(64, 32, concat=True, heads=4)
        self.bn3 = BatchNorm(128)
        self.conv4 = GATConv(128, 256, concat=True, heads=4)
        self.bn4 = BatchNorm(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn5 = BatchNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = BatchNorm(256)
        self.fc3 = nn.Linear(256, emb_size)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))  
        x = F.relu(self.bn4(self.conv4(x, edge_index)))

        x = global_max_pool(x, batch)
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        
        x = F.normalize(x, p=2, dim=1)
        return x

class GraphNetFeaturesPointNet(nn.Module):
    def __init__(self, num_node_features=1024, emb_size=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 1128)
        self.bn1 = BatchNorm(1128)
        self.conv2 = GCNConv(1128, 1256)
        self.bn2 = BatchNorm(1256)
        
        self.fc1 = nn.Linear(1256, 512)
        self.bn5 = BatchNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = BatchNorm(256)
        self.fc3 = nn.Linear(256, emb_size)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))

        x = global_max_pool(x, batch)
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        
        output = F.normalize(x, p=2, dim=1)

        return output
