# layopt
### Offical Code for Wind Farm Layout Optimization with Diffusion Models

### Environment Setup
```
    conda create -n layopt python=3.8 -y
    conda activate layopt

    # torch
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118  

    # floris
    pip install floris==4.1.1

    # torch_geometric
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
    pip install -r requirements.txt
```

### Execution
```
# Step 1. Data Preparation
python data/generate_data_diverse.py --num_turbins $num_turbins --layout_size $layout_size --num_samples $num_samples
```