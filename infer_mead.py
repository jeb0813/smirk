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

import json


def get_vids(dataset_json):
    vid_paths = []
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    for subject in dataset.keys():
        for emo in dataset[subject].keys():
            for level in dataset[subject][emo].keys():
                for vid in dataset[subject][emo][level].keys():
                    vid_paths.append(dataset[subject][emo][level][vid]["video"])
    return vid_paths

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

if __name__ == '__main__':
    import ipdb

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')

    parser.add_argument('--dataset_json', type=str, default='dataset.json', help='Path to the dataset json file')

    args = parser.parse_args()

    input_image_size = 224


    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()


    vid_paths = get_vids(args.dataset_json)
    for vid in vid_paths:
        print(f'Processing {vid}')
        result = list()

        cap = cv2.VideoCapture(vid)

        if not cap.isOpened():
            print('Error opening video file')
            exit()

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cnt=0
        while True:
            ret, image = cap.read()

            if not ret:
                break
            cnt+=1
            kpt_mediapipe = run_mediapipe(image)

            # crop face if needed
            if args.crop:
                if (kpt_mediapipe is None):
                    print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                    exit()
                
                kpt_mediapipe = kpt_mediapipe[..., :2]

                tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=input_image_size)
                
                cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

                cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
                cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
            else:
                cropped_image = image
                cropped_kpt_mediapipe = kpt_mediapipe

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, (224,224))
            cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
            cropped_image = cropped_image.to(args.device)

            outputs = smirk_encoder(cropped_image)
            outputs_numpy = {k:v.cpu().detach().numpy() for k,v in outputs.items()}
            tform_matrix = tform.params
            outputs_numpy['tform'] = tform_matrix
            result.append(outputs_numpy)

        assert cnt == len(result)
        # ipdb.set_trace()
        param_path = vid.replace('video','param').replace('.mp4','')
        if not os.path.exists(param_path):
            os.makedirs(param_path)
        
        for i,outputs_numpy in enumerate(result):
            # 保存为npz
            np.savez(f"{param_path}/{i+1:03d}.npz", **outputs_numpy)

