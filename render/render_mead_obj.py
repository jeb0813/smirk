import sys
sys.path.append('/data/chenziang/codes/smirk')

import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer, RendererObj
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F

from skimage.transform import SimilarityTransform

import trimesh

if __name__ == "__main__":
    import ipdb
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--param_path', type=str, default='MEAD/M003/param/angry/level_1/001')
    parser.add_argument('--vid_path', type=str, default='MEAD/M003/video/angry/level_1/001.mp4')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    
    args = parser.parse_args()


    # ---- visualize the results ---- #
    flame = FLAME().to(args.device)
    # renderer = Renderer().to(args.device)
    renderer = RendererObj(render_full_head=False).to(args.device)

    cap = cv2.VideoCapture(args.vid_path)

    if not cap.isOpened():
        print('Error opening video file')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = video_width
    out_height = video_height
    out_width *= 2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    cap_out = cv2.VideoWriter(f"{args.out_path}/{args.vid_path.split('/')[-1].split('.')[0]}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (out_width, out_height))

    cnt=0
    while True:
        cnt+=1
        print('frame:', cnt)
        ret, image = cap.read()

        if not ret:
            break
        
        param_path = os.path.join(args.param_path, f'{cnt:03d}.npz')
        param = np.load(param_path)

        if cnt==1:
            tform_matrix = param['tform']
            tform = SimilarityTransform(matrix=tform_matrix)

        cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(args.device)

        output_params = ['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params']
        outputs = {k:torch.tensor(param[k]).to(args.device) for k in output_params}
        # set pose to zero
        outputs['pose_params'] = torch.zeros_like(outputs['pose_params'])
        # share cam params
        if cnt==1:
            cam = outputs['cam']

        # import ipdb; ipdb.set_trace()
        # vertices [1, 5023, 3]
        flame_output = flame.forward(outputs)

        # temp = flame_output['vertices'].clone().squeeze(0)
        # temp = temp.cpu().detach().numpy()
        
        # # temp保存为txt
        # np.savetxt(f'results/vertices_{cnt}.txt', temp)

        renderer_output = renderer.forward(flame_output['vertices'], cam,
                                            landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        # [1, 3, 224, 224]
        rendered_img = renderer_output['rendered_img']

        # ipdb.set_trace()

        rendered_img_numpy = (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
        rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)

        # back to pytorch to concatenate with full_image
        rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0

        full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
        grid = torch.cat([full_image, rendered_img_orig], dim=3)

        grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

        # cv2.imwrite(f'results/output_{cnt}.png', grid_numpy)

        cap_out.write(grid_numpy)
    
    cap.release()
    cap_out.release()

