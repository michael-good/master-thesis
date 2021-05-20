import os
import glob
import h5py
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import KNNGraph

class GraphDataset(InMemoryDataset):
    def __init__(self, root, num_files=86, k=9, undirected=False):
        super(GraphDataset, self).__init__(root, transform=KNNGraph(k=k, loop=True, force_undirected=undirected), pre_transform=None)
        self.num_files = num_files
        self.flist = os.path.join(root, "*.h5")
        self.flist = sorted(glob.glob(self.flist))
        self.graphs = []
        self.images = []
        self.load_data()
                
    def load_data(self):
        files = [h5py.File(fname, 'r') for fname in self.flist[:self.num_files]]
        point_clouds = []
        for f in files:
            point_clouds.append(f['points'][:])
            self.images.append(f['images'][:])
            print("> File {} loaded ...".format(f), flush=True)
        point_clouds = torch.from_numpy(np.concatenate(np.array(point_clouds)))
        
        for pc in point_clouds:
            graph = Data(x=pc)
            graph = self.transform(graph)
            self.graphs.append(graph)

        self.images = torch.from_numpy(np.concatenate(np.array(self.images)))
        self.images = self.images.permute(0, 3, 1, 2)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):        
        return self.graphs[i], self.images[i]
