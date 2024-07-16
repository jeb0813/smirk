import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F

from skimage.transform import SimilarityTransform

if __name__ == "__main__":
    import ipdb
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--param_path', type=str, default='/data/chenziang/codes/Mimic/HDTF-3D/spectre_processed_25fps_16kHz/RD_Radio1_000/verts_new_shape1.npy')
    # parser.add_argument('--vid_path', type=str, default='MEAD/M003/video/angry/level_1/001.mp4')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    
    args = parser.parse_args()
    
    # ---- visualize the results ---- #
    flame = FLAME().to(args.device)
    renderer = Renderer(render_full_head=True).to(args.device)

    video_fps = 25
    video_width = 512
    video_height = 512
    cap_out = cv2.VideoWriter(f"{args.out_path}/{args.param_path.split('/')[-2]}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))


    # 读取param
    verts = np.load(args.param_path, allow_pickle=True)[()]['verts']
    verts = torch.tensor(verts).float().to(args.device)
    n = verts.shape[0]
    
    
    # cam param shape (1, 3)
    cam = torch.tensor([9.0, 0.0, 0.0]).unsqueeze(0).to(args.device)

    for i in range(n):
        print('frame:', i+1)
        
        vert = verts[i].unsqueeze(0)
        renderer_output = renderer.forward(vert, cam)

        rendered_img = renderer_output['rendered_img']
        rendered_img_numpy = (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
        
        cap_out.write(rendered_img_numpy)


    cap_out.release()

