import sys
sys.path.append('/data/chenziang/codes/smirk')

from src.FLAME.FLAME import FLAME

import os
import torch
import numpy as np


if __name__ == "__main__":
    template_path = '/data/chenziang/codes/smirk/templates'

    device = 'cuda'
    flame = FLAME().to(device)


    for file in sorted(os.listdir(template_path)):
        if len(file) < 8:
            continue
        param_path = os.path.join(template_path, file)
        param = np.load(param_path)
        
        param = {key: torch.tensor(param[key]).squeeze(1).to(device) for key in param.keys()}

        vertices = flame.forward(param)['vertices']
        vertices_np = vertices.squeeze(0).cpu().detach().numpy()

        save_path = param_path.replace('_no_eyelid.npz', '.npy')
        # ipdb.set_trace()

        np.save(save_path, vertices_np)
