import sys
sys.path.append('/data/chenziang/codes/smirk')

from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer

import os
import json
import torch

import numpy as np

def load_dataset(dataset_json):
    ret=list()
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    for subject in dataset:
        for emo in dataset[subject]:
            for level in dataset[subject][emo]:
                for vid in dataset[subject][emo][level]:
                    ret.append(dataset[subject][emo][level][vid]["param"])

    return ret


if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    raw_json = '/data/chenziang/codes/smirk/index.json'
    device = 'cuda'
    params = load_dataset(raw_json)

    flame = FLAME().to(device)
    # renderer = Renderer(render_full_head=True, obj_filename='/data/chenziang/codes/smirk/assets/head_template.obj').to(device)

    for param in params:
        print('Processing:', param)
        _param = np.load(param)
        _param = {key: torch.tensor(_param[key]).squeeze(1).to(device) for key in _param.keys()}
        # set pose to zero
        _param['pose_params'] = torch.zeros_like(_param['pose_params'])
        # set eyelid to zero
        _param['eyelid_params'] = torch.zeros_like(_param['eyelid_params'])

        vertices = flame.forward(_param)['vertices']
        vertices_np = vertices.squeeze(0).cpu().detach().numpy()

        save_path = param.replace('param.npz', 'vertices.npy')
        # ipdb.set_trace()

        np.save(save_path, vertices_np)
        # exit()


