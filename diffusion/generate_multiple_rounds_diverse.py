import os
import random
import pickle
import argparse
import pathlib

import gin
import numpy as np
from tqdm import tqdm
import torch
import wandb

from torch.utils.data import Dataset, DataLoader
from data.dataset import WindFarmLayoutDataset, GraphWindFarmLayoutDataset
from diffusion.elucidated_diffusion import Trainer
from diffusion.norm import MinMaxNormalizer
from diffusion.utils import construct_diffusion_model

from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch

import data.generation_utils as gu
import multiprocessing as mp

class GraphSimpleDiffusionGenerator:
    def __init__(
            self,
            graphs,
            ema_model,
            num_turbins: int = 20,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100,
            guidance_scale: float = 1.0,
            device: torch.device = torch.device('cuda'),
    ):
        self.graphs = graphs
        
        self.diffusion = ema_model
        self.diffusion.eval()
        
        self.num_turbins = num_turbins
        self.device = device
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        self.guidance_scale = guidance_scale
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')
        
    def sample(
            self,
            num_samples: int,
            cond: torch.Tensor,
            add_cond: torch.Tensor,
    ) -> np.ndarray:
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        
        xs = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            batch = GraphBatch.from_data_list(self.graphs[i*self.sample_batch_size: (i+1)*self.sample_batch_size]).to(self.device)
            
            sampled_outputs = self.diffusion.sample(
                x=batch.x,
                adj=batch.edge_index,
                ef=batch.edge_attr,
                num_turbins=self.num_turbins,
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond=cond[i*self.sample_batch_size: (i+1)*self.sample_batch_size] if cond is not None else None,
                temperature=self.guidance_scale,
                add_cond=add_cond[i*self.sample_batch_size: (i+1)*self.sample_batch_size],
            )
            sampled_outputs = sampled_outputs
            xs.append(sampled_outputs)

        xs = torch.cat(xs, dim=0)
        return xs

class SimpleDiffusionGenerator:
    def __init__(
            self,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100,
            guidance_scale: float = 1.0,
    ):
        self.diffusion = ema_model
        self.diffusion.eval()
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        self.guidance_scale = guidance_scale
        
    def sample(
            self,
            num_samples: int,
            cond: torch.Tensor,
            add_cond: torch.Tensor,
    ) -> np.ndarray:
        assert num_samples % self.sample_batch_size == 0
        num_batches = num_samples // self.sample_batch_size
        
        xs = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond=cond[i*self.sample_batch_size: (i+1)*self.sample_batch_size] if cond is not None else None,
                temperature=self.guidance_scale,
                add_cond=add_cond,
            )
            xs.append(sampled_outputs)

        xs = torch.cat(xs, dim=0)
        return xs
    
def main():
    parser = argparse.ArgumentParser()
    # Dataset config
    parser.add_argument('--num_turbins', type=int, default=10)
    parser.add_argument('--layout_size', type=float, default=2000.0)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--data_type', type=str, default='aep')
    # Divserse wind scenarios
    parser.add_argument("--wind_direction_min", type=float, default=0.0)
    parser.add_argument("--wind_direction_max", type=float, default=120.0)
    parser.add_argument("--wind_speed_min", type=float, default=6.0)
    parser.add_argument("--wind_speed_max", type=float, default=10.0)
    parser.add_argument("--wind_direction_target", type=float, default=60.0)
    parser.add_argument("--wind_speed_target", type=float, default=8.0)
    
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--normalizer_type', type=str, default='minmax')
    
    # Bootstrap config
    parser.add_argument('--num_cycles', type=int, default=10)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--guidance_multiplier', type=float, default=1.0)
    
    # Training config
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--small_batch_size', type=int, default=256)
    parser.add_argument('--train_num_steps', type=int, default=5000)
    # parser.add_argument('--train_num_steps_boot', type=int, default=1000)
    # parser.add_argument('--load_train_num_steps', type=int, default=10000)
    parser.add_argument('--num_generated_samples', type=int, default=1000)
    parser.add_argument('--save_and_sample_every', type=int, default=10000)
    parser.add_argument('--train_lr', type=float, default=1e-4)
    parser.add_argument('--cond_type', type=str, default="value")
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    
    # Other config
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--wandb-project', type=str, default="layout_optimization_bootstrap")
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    
    # Load initial dataset
    data_dir = f"Experiments/dataset/{args.num_turbins}_{args.num_samples}_{args.layout_size}x{args.layout_size}_{args.data_type}"
    data_dir += f"_speed{args.wind_speed_min}_{args.wind_speed_max}_direction{args.wind_direction_min}_{args.wind_direction_max}.pkl"
    data = np.load(data_dir, allow_pickle=True)
    
    layouts, scores, wind_scenarios = data["layouts"], data["scores"], data["wind_scenario"]
    layouts = torch.from_numpy(layouts).float()
    scores = torch.from_numpy(scores).float().unsqueeze(1)
    
    wind_speeds = np.array([wind_scenario.wind_speeds[0] / 10.0 for wind_scenario in wind_scenarios])
    wind_directions = np.array([wind_scenario.wind_directions[0] / 180.0 for wind_scenario in wind_scenarios])
    wind_scenarios = torch.from_numpy(np.stack([wind_speeds, wind_directions], axis=1)).float()
    
    # Create the dataset
    if args.model_type == 'mlp':
        dataset = WindFarmLayoutDataset(layouts, scores, wind_scenarios)
    elif args.model_type == 'gnn': 
        dataset = GraphWindFarmLayoutDataset(layouts, scores, wind_scenarios)
    
    wandb.init(
        project=args.wandb_project,
        name=f"diffusion_{args.num_turbins}_{args.num_samples}_{args.layout_size}",
        config=vars(args),
    )
    
    # Create initial diffusion model
    device = torch.device("cuda" if args.use_gpu else "cpu")
    diffusion_model = construct_diffusion_model(
        inputs=layouts,
        model_type=args.model_type,
        normalizer_type=args.normalizer_type,
        cond_dim=1,
        add_cond_dim=2,
    ).to(device)
    
    train_results_dir = f"Experiments/diffusion_{args.model_type}_{args.data_type}"
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)
        
    train_results_path = os.path.join(
        train_results_dir, 
        f'{args.num_turbins}_{args.num_samples}_{args.layout_size}x{args.layout_size}'
    )
    train_results_path += f"_speed{args.wind_speed_min}_{args.wind_speed_max}_direction{args.wind_direction_min}_{args.wind_direction_max}"
    
    print("Training model...")
    trainer = Trainer(
        diffusion_model=diffusion_model,  # Continue training the same model
        dataset=dataset,
        train_batch_size=args.train_batch_size,
        small_batch_size=args.small_batch_size,
        train_num_steps=args.train_num_steps,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=train_results_path,
        cond_type=args.cond_type,
        cond_drop_prob=args.cond_drop_prob,
        add_cond=True,
    )
    trainer.train()    
        
    sample_results_dir = f"Experiments/sample_{args.model_type}_{args.data_type}"
    if not os.path.exists(sample_results_dir):
        os.makedirs(sample_results_dir)
        
    sample_results_path = os.path.join(sample_results_dir, f'{args.guidance_scale}_{args.guidance_multiplier}_{args.num_turbins}_{args.num_samples}_{args.layout_size}x{args.layout_size}')
    sample_results_path += f"_target_speed{args.wind_speed_target}_target_direction{args.wind_direction_target}_epoch{args.train_num_steps}"
    
    # Main loop
    for cycle in range(args.num_cycles):
        print(f"\nCycle {cycle+1}/{args.num_cycles}")
        
        train_results_path = os.path.join(
            train_results_dir, 
            f'{args.num_turbins}_{args.num_samples}_{args.layout_size}x{args.layout_size}'
        )
        train_results_path += f"_speed{args.wind_speed_min}_{args.wind_speed_max}_direction{args.wind_direction_min}_{args.wind_direction_max}"
        
        # Train model on current dataset
        print("Training model...")
        trainer = Trainer(
            diffusion_model=diffusion_model,  # Continue training the same model
            dataset=dataset,
            train_batch_size=args.train_batch_size,
            small_batch_size=args.small_batch_size,
            train_num_steps=args.train_num_steps * 2,
            save_and_sample_every=args.save_and_sample_every,
            results_folder=train_results_path,
            cond_type=args.cond_type,
            cond_drop_prob=args.cond_drop_prob,
            add_cond=True,
        )
        trainer.ema.to(device)
        trainer.load(milestone=args.train_num_steps)
        trainer.train()
        
        # Generate samples using trained model
        print("Generating samples...")
        if args.model_type == 'mlp':
            generator = SimpleDiffusionGenerator(
                ema_model=trainer.ema.ema_model,
                guidance_scale=args.guidance_scale,
            )
        elif args.model_type == 'gnn':
            # Generate samples - Start from initial graph
            graphs = []
            idx = np.argsort(-scores.flatten())[:args.num_generated_samples]
            max_score = scores.max()
            print("Graph Processing...")
            for layout in layouts[idx]:
                # layout = torch.from_numpy(layout).float()
                layout_x, layout_y = layout[:,0], layout[:,1]
                u, v = [], []
                for i in range(args.num_turbins):
                    for j in range(args.num_turbins):
                        if i != j:
                            u.append(i)
                            v.append(j)
                edge_index = torch.tensor([u, v], dtype=torch.long)
                # mean_windspeed = wind_scenario.wind_speeds
                
                e_len = edge_index.shape[1]
                ef = torch.ones(e_len,1)
                
                add_cond = torch.tensor([[args.wind_speed_target / 10.0, args.wind_direction_target / 180.0]]).float()
                add_cond = add_cond.to(trainer.accelerator.device)
                graph = GraphData(x = torch.concat((layout_x.unsqueeze(1), layout_y.unsqueeze(1)), dim=1),
                            edge_index = edge_index,
                            edge_attr = ef.float(),
                            y = max_score.reshape(-1, 1) * args.guidance_multiplier,
                            add_cond = add_cond)
                graphs.append(graph)
            print("Processing Completed")
        
            generator = GraphSimpleDiffusionGenerator(
                graphs=graphs,
                ema_model=trainer.ema.ema_model,
                num_turbins=args.num_turbins,
                num_sample_steps=128,
                sample_batch_size=100,
                guidance_scale=args.guidance_scale,
                device=trainer.accelerator.device,
            )
            
        cond = torch.ones((args.num_generated_samples, 1)) * args.guidance_multiplier
        cond = cond.to(device)
        
        wind_scenario = gu.make_wind_scenario(wind_direction=args.wind_direction_target, wind_speed=args.wind_speed_target)
        wind_scenarios = [wind_scenario for _ in range(args.num_generated_samples)]
        add_cond = torch.tensor([[args.wind_speed_target / 10.0, args.wind_direction_target / 180.0]]).float()
        add_cond = add_cond.to(trainer.accelerator.device)
        add_cond = add_cond.repeat(args.num_generated_samples, 1)
        
        sampled_layouts = generator.sample(
            num_samples=args.num_generated_samples,
            cond=cond,
            add_cond=add_cond,
        )
        
        sampled_layouts = sampled_layouts.reshape(args.num_generated_samples, args.num_turbins, 2)
        sampled_layouts = sampled_layouts.cpu().detach().numpy()
        print(sampled_layouts[0, :4])
        # Filter violations
        min_dist = 2.0 * 126
        sampled_layouts = torch.tensor(sampled_layouts) 
        sampled_layouts = gu.filter_violations(
            n=args.num_turbins,
            max_x=args.layout_size,
            max_y=args.layout_size,
            min_d=min_dist,
            points=sampled_layouts,
        )
        print(f"Number of valid layouts: {len(sampled_layouts)}")
        
        # Evaluate samples
        print("Evaluating samples...")
        inputs = [(layout[:,0]*args.layout_size, layout[:,1]*args.layout_size, wind_scenario) 
                 for layout, wind_scenario in zip(sampled_layouts, wind_scenarios)]
        with mp.Pool(processes=5) as pool:
            if args.data_type == "aep":
                sampled_scores = pool.starmap(gu.compute_aep, tqdm(inputs, total=len(sampled_layouts)))
            elif args.data_type == "avp":
                sampled_scores = pool.starmap(gu.compute_avp, tqdm(inputs, total=len(sampled_layouts)))
        sampled_scores = np.array(sampled_scores)
        
        if cycle == 0:
            print(f"Dataset so far: {max(sampled_scores):.3f}")
            current_layouts = sampled_layouts
            current_scores = sampled_scores
            
            total_layouts = np.concatenate([layouts, sampled_layouts], axis=0)
            total_scores = np.concatenate([scores.flatten(), sampled_scores], axis=0)
        else:
            print(f"Dataset so far: {max(current_scores):.3f}\t Max in this round: {max(sampled_scores):.3f}")
            current_layouts = np.concatenate([current_layouts, sampled_layouts], axis=0)
            current_scores = np.concatenate([current_scores, sampled_scores], axis=0)
            
            total_layouts = np.concatenate([total_layouts, sampled_layouts], axis=0)
            total_scores = np.concatenate([total_scores, sampled_scores], axis=0)
        
        wind_speeds = np.array([wind_scenario.wind_speeds[0] / 10.0 for wind_scenario in wind_scenarios])
        wind_directions = np.array([wind_scenario.wind_directions[0] / 180.0 for wind_scenario in wind_scenarios])
        wind_scenarios = torch.from_numpy(np.stack([wind_speeds, wind_directions], axis=1)).float()
        
        wandb.log({
            "cycle": cycle + 1,
            "max_score_in_this_round": np.max(sampled_scores),
            "mean_score_in_this_round": np.mean(sampled_scores),
            "max_score_so_far": np.max(current_scores),
            "mean_score_so_far": np.mean(current_scores),
        })
        
        
        # Create new dataset for next cycle
        if args.model_type == 'mlp':
            dataset = WindFarmLayoutDataset(
                torch.from_numpy(current_layouts).float(),
                torch.from_numpy(current_scores).float().unsqueeze(1),
                wind_scenarios
            )
        elif args.model_type == 'gnn':
            dataset = GraphWindFarmLayoutDataset(
                torch.from_numpy(current_layouts).float(),
                torch.from_numpy(current_scores).float().unsqueeze(1),
                wind_scenarios
            )
        
        # Save results for this cycle
        cycle_results_path = f"{sample_results_path}_cycle{cycle}_seed{args.seed}_results.pkl"
        with open(cycle_results_path, 'wb') as f:
            pickle.dump({
                "layouts": current_layouts,
                "scores": current_scores,
                "sampled_layouts": sampled_layouts,
                "sampled_scores": sampled_scores,
                "wind_scenario": wind_scenario
            }, f)
        
if __name__ == '__main__':
    main()