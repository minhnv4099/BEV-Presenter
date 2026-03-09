import pickle
import json


with open('data/nuscenes/v1.0-mini/nuscenes_infos_temporal_train.pkl', 'rb') as f:
    print(type(pickle.loads(f.read())))
