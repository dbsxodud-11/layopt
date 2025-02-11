import os
import random
import pickle
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)

import generation_utils as gu
import multiprocessing as mp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_turbins", type=int, default=10)
    parser.add_argument("--layout_size", type=float, default=5000.0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--data_type", type=str, default="aep")
    
    args = parser.parse_args()
    
    # 1. Generate wind scenario
    wind_scenario = gu.make_wind_scenario()
    
    # 2. Generate random layouts
    points = gu.generate_points(batch=args.num_samples, 
                                n=args.num_turbins,
                                device="cpu",)

    # 3. Check constraints
    min_dist = 2.0*126
    points = points.detach()
    points = gu.filter_violations(n=args.num_turbins, 
                                    max_x=args.layout_size, 
                                    max_y=args.layout_size,
                                    min_d=min_dist,
                                    points=points)
        
    print(f"Layout detail : {args.num_turbins} windturbines in {args.layout_size}x{args.layout_size}")
    print(f"Number of layouts after filtering: {args.num_samples} -> {len(points)}")

    
    # 4. Calculate AEP of wind farm layouts
    # scores = []
    # for layout in tqdm(points):
    #     layout_x, layout_y = layout[:,0], layout[:,1]
    #     score = gu.compute_aep(layout_x*args.layout_size, layout_y*args.layout_size, wind_scenario)
    #     scores.append(score)
    # scores = np.array(scores)
    
    # multiprocessing (much faster)
    inputs = [(layout[:,0]*args.layout_size, layout[:,1]*args.layout_size, wind_scenario) for layout in points]
    with mp.Pool(processes=5) as pool:
        if args.data_type == "aep":
            scores = pool.starmap(gu.compute_aep, tqdm(inputs, total=len(points)))
        elif args.data_type == "avp":
            scores = pool.starmap(gu.compute_avp, tqdm(inputs, total=len(points)))
    pool.close()
    pool.join()
    scores = np.array(scores)
    
    # 3. Save the dataset
    if not os.path.exists("Experiments/dataset"):
        os.makedirs("Experiments/dataset", exist_ok=True)
    with open(f"Experiments/dataset/{args.num_turbins}_{args.num_samples}_{args.layout_size}x{args.layout_size}_{args.data_type}.pkl", "wb") as f:
        pickle.dump({"layouts": points, "scores": scores, "wind_scenario": wind_scenario}, f)
    
        