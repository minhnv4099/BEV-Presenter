# The project of COMP-5421: BEV Presenter.


## Create virtual environment
```bash
uv venv --python 3.10.0
```
## Install dependencies
```bash
uv pip install -e .
```

## Token
Get token [here](https://docs.google.com/document/d/129TDtn83w0sZky860JnfVmeleP7Bv74rjDI1J3dz83M/edit?usp=sharing).
Then paste to `~/.cache/huggingface/token` file (in Linux) or set environment variable ``HF_TOKEN`` to access [organization repository](https://huggingface.co/5421Project). 

## Prepare data
Run command to download can bus and nuscene data from [repo](https://huggingface.co/datasets/5421Project/nuscene).
```bash
uv run src/tools/download_data.py \
    --repo 5421Project/nuscene \
    --out-dir data \
    --version v1.0-mini \
    --flag nuscenes
```
From here, can visualize data sample (see [visualize.ipynb](notebooks/visualize.ipynb)).

Run command to prepare metadata

```bash
 uv run src/tools/create_data.py nuscenes \
    --root-path data/nuscenes \
    --version v1.0-mini \
    --extra-tag nuscenes \
    --canbus data/nuscenes/can_bus 
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`
in ``data/nuscenes/v1.0-mini/``.

Folder structure after downloading and preparing data looks like
```
bevformer/
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── v1.0-mini/
│   │   │   ├── maps/
│   │   │   ├── reverse/
│   │   │   ├── samples/
│   │   │   ├── sweeps/
│   │   │   ├── v1.0-mini/
│   |   |   ├── nuscenes_infos_temporal_train.json
│   |   |   ├── nuscenes_infos_temporal_train.pkl
│   |   |   ├── nuscenes_infos_temporal_val.json
│   |   |   ├── nuscenes_infos_temporal_val.pkl
```


## Train
Run below command to see training instructions:
```bash
uv run train.py --help
```

Then train by, can leave everything default:
```bash
uv run train.py \
   --config configs/bevformer_tiny_test.py \
   --work-dir experiment \
   --experiment-name baseline
```

Checkpoints are pushed to repo **[5421Project](https://huggingface.co/datasets/5421Project)/{experiment_name}** intervally.


## Config
See [bevformer_tiny_test.py](configs/bevformer_tiny_test.py) to understand config and edit if needed.
