import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData


class WindFarmLayoutDataset(Dataset):
    def __init__(self, layouts, scores, wind_scenarios=None):
        self.layouts = layouts
        self.scores = scores
        
        if wind_scenarios is not None:
            self.wind_scenarios = wind_scenarios
        
        self.layouts = self.layouts.flatten(1)
        self.scores = (self.scores - self.scores.min()) / (self.scores.max() - self.scores.min() + 1e-7)
        self.scores = self.scores
        
    def __len__(self):
        return len(self.layouts)
    
    def __getitem__(self, idx):
        layout = self.layouts[idx]
        score = self.scores[idx]
        if hasattr(self, 'wind_scenarios'):
            wind_scenario = self.wind_scenarios[idx]
            return layout, score, wind_scenario
        else:
            return layout, score
    

class GraphWindFarmLayoutDataset(Dataset):
    def __init__(self, layouts, scores, wind_scenarios=None):
        self.layouts = layouts
        self.scores = scores
        
        if wind_scenarios is not None:
            self.wind_scenarios = wind_scenarios
        
        self.scores = (self.scores - self.scores.min()) / (self.scores.max() - self.scores.min() + 1e-7)
        self.datalist = []
        num_turbins = layouts.shape[1]
        print("Graph Processing...")
        if wind_scenarios is not None:
            for layout, score, wind_scenario in zip(tqdm(self.layouts), self.scores, self.wind_scenarios):
                layout_x, layout_y = layout[:,0], layout[:,1]
                u, v = [], []
                for i in range(num_turbins):
                    for j in range(num_turbins):
                        if i != j:
                            u.append(i)
                            v.append(j)
                edge_index = torch.tensor([u, v], dtype=torch.long)
                
                e_len = edge_index.shape[1]
                # Fix Edge Attribute here
                ef = torch.ones(e_len,1)
                add_cond = wind_scenario.unsqueeze(0)

                graph = GraphData(x = torch.concat((layout_x.unsqueeze(1), layout_y.unsqueeze(1)), dim=1),
                            edge_index = edge_index,
                            edge_attr = ef.float(),
                            y = score.reshape(-1, 1),
                            add_cond = add_cond)
                self.datalist.append(graph)
        else:
            for layout, score in zip(tqdm(self.layouts), self.scores):
                layout_x, layout_y = layout[:,0], layout[:,1]
                u, v = [], []
                for i in range(num_turbins):
                    for j in range(num_turbins):
                        if i != j:
                            u.append(i)
                            v.append(j)
                edge_index = torch.tensor([u, v], dtype=torch.long)
                e_len = edge_index.shape[1]
                ef = torch.ones(e_len,1)
                graph = GraphData(x = torch.concat((layout_x.unsqueeze(1), layout_y.unsqueeze(1)), dim=1),
                            edge_index = edge_index,
                            edge_attr = ef.float(),
                            y = score.reshape(-1, 1))
                self.datalist.append(graph)
            
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        return self.datalist[idx]