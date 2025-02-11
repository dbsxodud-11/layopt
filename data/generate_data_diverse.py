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
    # Divserse wind scenarios
    parser.add_argument("--wind_direction_min", type=float, default=0.0)
    parser.add_argument("--wind_direction_max", type=float, default=120.0)
    parser.add_argument("--wind_speed_min", type=float, default=6.0)
    parser.add_argument("--wind_speed_max", type=float, default=10.0)
    
    args = parser.parse_args()
    
    # 1. Generate wind scenario
    wind_speeds = np.random.uniform(args.wind_speed_min, args.wind_speed_max, args.num_samples)
    wind_directions = np.random.uniform(args.wind_direction_min, args.wind_direction_max, args.num_samples)
    wind_scenarios = [gu.make_wind_scenario(wind_speed=wind_speed, wind_direction=wind_direction) for wind_speed, wind_direction in zip(wind_speeds, wind_directions)]
    
    # 2. Generate layouts
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
    inputs = [(layout[:,0]*args.layout_size, layout[:,1]*args.layout_size, wind_scenario) for layout, wind_scenario in zip(points, wind_scenarios)]
    with mp.Pool(processes=5) as pool:
        if args.data_type == "aep":
            scores = pool.starmap(gu.compute_aep, tqdm(inputs, total=len(inputs)))
        elif args.data_type == "avp":
            scores = pool.starmap(gu.compute_avp, tqdm(inputs, total=len(inputs)))
    pool.close()
    pool.join()
    scores = np.array(scores)
    
    # 3. Save the dataset
    if not os.path.exists("Experiments/dataset"):
        os.makedirs("Experiments/dataset", exist_ok=True)
    file_name = f"Experiments/dataset/{args.num_turbins}_{args.num_samples}_{args.layout_size}x{args.layout_size}_{args.data_type}"
    file_name += f"_speed{args.wind_speed_min}_{args.wind_speed_max}_direction{args.wind_direction_min}_{args.wind_direction_max}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump({"layouts": points, "scores": scores, "wind_scenario": wind_scenarios}, f)
    
        