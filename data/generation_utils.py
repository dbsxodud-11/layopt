import torch
import numpy as np
from tqdm import tqdm
from floris import FlorisModel, WindRose


def make_wind_scenario(wind_direction = 60.0, wind_speed = 8.0):
    """
    Make a wind speed scenario
    :param wind_direction_bin: (float) The size of the wind direction bins. in angle.
    :param wind_speed_bin: (float) The size of the wind speed bins.
    :param shape_boundary: (list of floats) The lower and upper bounds for the shape parameter.
    :param scale_boundary: (list of floats) The lower and upper bounds for the scale parameter.
    """
    # wind_directions = np.arange(0, 360.0, wind_direction_bin)
    wind_directions = np.array([wind_direction])
    wind_speeds = np.array([wind_speed])
    
    # Shape random frequency distribution to match number of wind directions and wind speeds
    freq_table = np.zeros((len(wind_directions), len(wind_speeds)))
    np.random.seed(1)
    freq_table[:,0] = (np.abs(np.sort(np.random.randn(len(wind_directions)))))
    freq_table = freq_table / freq_table.sum()

    # Define the value table such that the value of the energy produced is
    # significantly higher when the wind direction is close to the north or
    # south, and zero when the wind is from the east or west. Here, value is
    # given a mean value of 25 USD/MWh.
    value_table = (0.5 + 0.5*np.cos(np.radians(wind_directions)))**10
    value_table = 25*value_table/np.mean(value_table)
    value_table = value_table.reshape((len(wind_directions),1))
    
    return WindRose(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        freq_table=freq_table,
        ti_table=0.06,
        value_table=value_table,        
    )


def generate_points(batch, n, device):
    pts = torch.rand(batch, n, 2, device=device, requires_grad=True)
    return pts


# function to calculate the minimum distance between all pairs of points
def calc_dists(points, max_x, max_y):
    x,y = points[:,:,0].unsqueeze(2), points[:,:,1].unsqueeze(2)
    scld = torch.concat((x*max_x, y*max_y), dim=2)
    return torch.cdist(scld, scld, p=2)

# function to optimize the positions of the points with respect to a given minimum distance constraint
def optimize_points(batch, n, min_dist, max_x, max_y,
                    lr=0.001, num_iters=1000, device='cuda'):
    # generate initial set of random points
    pts = generate_points(batch, n, device=device)

    # create Adam optimizer
    optimizer = torch.optim.Adam([pts], lr=lr)

    # iterate optimization for a fixed number of iterations
    for i in tqdm(range(num_iters)):
        # calculate distances between all pairs of points
        dists = calc_dists(pts, max_x, max_y)
        loss = torch.sum(min_dist - dists[dists < min_dist])

        x_min_mask = pts[..., 0] <= 0
        x_max_mask = pts[..., 0] >= 1
        x_loss = torch.sum(0 - pts[x_min_mask]) + torch.sum(pts[x_max_mask] - 1)

        y_min_mask = pts[..., 1] <= 0
        y_max_mask = pts[..., 1] >= 1
        y_loss = torch.sum(0 - pts[y_min_mask]) + torch.sum(pts[y_max_mask] - 1)
        
        loss = loss + x_loss + y_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pts = torch.clip(pts, 0,1)
    return pts

# function to optimize the positions of the points with respect to a given minimum distance constraint
def optimize_points_sample(pts, n, min_dist, max_x, max_y,
                    lr=0.001, num_iters=1000, device='cuda'):
    # generate initial set of random points
    # pts = generate_points(batch, n, device=device)
    pts.requires_grad = True
    
    # create Adam optimizer
    optimizer = torch.optim.Adam([pts], lr=lr)

    # iterate optimization for a fixed number of iterations
    for i in tqdm(range(num_iters)):
        # calculate distances between all pairs of points
        dists = calc_dists(pts, max_x, max_y)
        loss = torch.sum(min_dist - dists[dists < min_dist])

        x_min_mask = pts[..., 0] <= 0
        x_max_mask = pts[..., 0] >= 1
        x_loss = torch.sum(0 - pts[x_min_mask]) + torch.sum(pts[x_max_mask] - 1)

        y_min_mask = pts[..., 1] <= 0
        y_max_mask = pts[..., 1] >= 1
        y_loss = torch.sum(0 - pts[y_min_mask]) + torch.sum(pts[y_max_mask] - 1)
        
        loss = loss + x_loss + y_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pts = torch.clip(pts, 0,1)
    return pts

def filter_violations(n, max_x, max_y, min_d, points):
    mask = (calc_dists(points, max_x, max_y) <= min_d).sum(dim=(1,2))-n
    points_unmask = points[mask==0]
    points_mask = optimize_points_sample(points[mask!=0], len(mask), min_d, max_x, max_y)
    # points = points[mask==0]
    points = torch.cat((points_unmask, points_mask), dim=0)
    
    mask = (calc_dists(points, max_x, max_y) <= min_d).sum(dim=(1,2))-n
    points = points[mask==0]
    return points.detach().cpu().numpy()


# @ray.remote(num_gpus=2)
def compute_aep(layout_x,
                layout_y,
                wind_scenario):
    """
    Compute AEP and each turbines' aep about random layout without optimization.
    :param initial_layout: (dict) The layout of the windfarm.
    :param wind_scenario: (dict) The wind scenario.
    """
    layout_x = layout_x.tolist()
    layout_y = layout_y.tolist()
    fmodel = FlorisModel("./config/gch.yaml")
    fmodel.set(wind_data=wind_scenario,
           layout_x=layout_x,
           layout_y=layout_y)
    
    fmodel.run()
    turbine_power = fmodel.get_farm_AEP() / 1e6
    return turbine_power