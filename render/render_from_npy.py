import sys
sys.path.append('/data/chenziang/codes/smirk')

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

import trimesh

import tempfile
from subprocess import call


if __name__ == "__main__":
    import ipdb
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--npy_path', type=str, default='MEAD/M003/param/angry/level_1/001/vertices.npy')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--img_size', type=int, default=512)
    
    args = parser.parse_args()

    
    renderer = Renderer(render_full_head=True, image_size=args.img_size).to(args.device)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    vertices = np.load(args.npy_path)
    vertices = torch.tensor(vertices).to(args.device)
    n = vertices.shape[0]

    cam = torch.tensor([8, 0, 0]).unsqueeze(0).repeat(n, 1).to(args.device)

    
    renderer_output = renderer.forward(vertices, cam)
    # [1, 3, 224, 224] [98, 3, 224, 224]
    rendered_img = renderer_output['rendered_img']

    rendered_img_numpy = (rendered_img.permute(0,2,3,1).detach().cpu().numpy()*255.0).astype(np.uint8)           


    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=args.out_path)
    # cap_out = cv2.VideoWriter(f"{args.out_path}/{'_'.join(args.npy_path.split('.')[0].split('/'))}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), args.video_fps, (args.img_size, args.img_size))
    cap_out = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.video_fps, (args.img_size, args.img_size), True)

    ipdb.set_trace()

    for i in range(n):
        cap_out.write(rendered_img_numpy[i])

    
    cap_out.release()

    # ffmpeg 转码
    file_name = '_'.join(args.npy_path.split('.')[0].split('/'))
    video_fname = os.path.join(args.out_path, file_name+'-no_audio.mp4')
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(tmp_video_file.name, video_fname)).split()
    call(cmd)

    wav_file = args.npy_path.replace('param','audio').replace('/vertices.npy', '')+'.m4a'

    # add audio
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -qscale 0 {2}'.format(wav_file, video_fname, video_fname.replace('-no_audio.mp4', '.mp4'))).split()
    call(cmd)
    if os.path.exists(video_fname):
        os.remove(video_fname)


