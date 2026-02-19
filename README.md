# The project of COMP-5421: BEV Presenter.


# Installation
Use **uv** to install dependencies in [uv.lock](uv.lock). (need to make compatible)

## NOTE
Should run any file by ```uv run ...```.

## Prepare data
**Download nuScenes data and CAN bus expansion**

Download nuScenes data (**mini version**) and CAN bus expansion data [HERE](https://www.nuscenes.org/nuscenes), extract and
then place them in [`data/nuscenes/`](data/nuscenes).

**Prepare nuScenes data**

Run command to prepare data

*We genetate custom annotation files which are different from mmdet3d's*

```bash
 uv run src/tools/create_data.py nuscenes \
        --root-path data/nuscenes \
        --version v1.0-mini \
        --extra-tag nuscenes \
        --canbus data/nuscenes/can_bus 
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`
in ``data/nuscenes/v1.0-mini/``.

**Folder structure**
```
BEVFormer
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── v1.0-mini/
│   │   │   ├── maps/
│   │   │   ├── samples/
│   │   │   ├── sweeps/
│   │   │   ├── v1.0-mini/
│   |   |   ├── nuscenes_infos_temporal_train.pkl
│   |   |   ├── nuscenes_infos_temporal_val.pkl
```

## Train
```bash
uv run train.py \
   --config configs/bevformer_tiny_test.py \
   --work-dir experiment/ \
   --experiment-name train
```

Checkpoints are pushed to https://huggingface.co/5421Project/bevformer.

Paste token to `~/.cache/huggingface/token` to access the above repo. 

## Config
See [bevformer_tiny_test.py](configs/bevformer_tiny_test.py) to edit config and hyperparameters.