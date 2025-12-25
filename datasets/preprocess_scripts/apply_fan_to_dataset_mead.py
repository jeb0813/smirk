from tqdm import tqdm
import numpy as np
import os 
import cv2
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
import argparse
from ibug.face_alignment.utils import plot_landmarks

from multiprocessing import Pool

import ipdb

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process images/videos with https://github.com/hhj1897/face_alignment.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
parser.add_argument('--vis_dir', type=str, help='Directory to save visualizations')
parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use for processing')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for detectors, e.g., cuda:0 or cpu')
args = parser.parse_args()


all_files = []

for subject_id in os.listdir(args.input_dir):
    subject_dir = os.path.join(args.input_dir, subject_id)
    if not os.path.isdir(subject_dir):
        continue

    # 每个被试的 video 目录
    video_root = os.path.join(subject_dir, "video")
    if not os.path.isdir(video_root):
        continue

    # 遍历 emotion（angry, happy, ...）
    for emotion in os.listdir(video_root):
        emotion_dir = os.path.join(video_root, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        # 遍历 level（level_1, level_2, level_3）
        for level in os.listdir(emotion_dir):
            level_dir = os.path.join(emotion_dir, level)
            if not os.path.isdir(level_dir):
                continue

            # 最后才是具体的 mp4 文件
            for file_name in os.listdir(level_dir):
                if file_name.lower().endswith(('.mp4', '.avi')):
                    all_files.append((level_dir, file_name))


# 每个进程内各自初始化一次（全局变量存在于子进程中）
_face_detector = None
_landmark_detector = None
_input_dir = None
_output_dir = None

def init_worker(input_dir, output_dir, device):
    """每个 worker 启动时执行一次：初始化模型 + 保存全局路径配置。"""
    global _face_detector, _landmark_detector, _input_dir, _output_dir
    _input_dir = input_dir
    _output_dir = output_dir

    _face_detector = RetinaFacePredictor(
        threshold=0.8, device=device,
        model=RetinaFacePredictor.get_model('mobilenet0.25')
    )
    _landmark_detector = FANPredictor(
        device=device, model=FANPredictor.get_model('2dfan2_alt')
    )

def process_one_video(item):
    """处理一个视频：逐帧提取 landmarks，保存为一个 npy。"""
    global _face_detector, _landmark_detector, _input_dir, _output_dir

    root, file_name = item
    input_path = os.path.join(root, file_name)

    rel_path = os.path.relpath(input_path, _input_dir)
    output_path = os.path.join(_output_dir, os.path.splitext(rel_path)[0] + '.npy')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return ("open_failed", input_path)

    all_landmarks = []

    while True:
        ret, image = cap.read()
        if not ret:
            break

        detected_faces = _face_detector(image, rgb=False)
        landmarks, scores = _landmark_detector(image, detected_faces, rgb=False)

        all_landmarks.append(landmarks)

    cap.release()

    np.save(output_path, np.array(all_landmarks, dtype=object), allow_pickle=True)
    return ("ok", input_path)

def main():
    # NOTE: 单卡 GPU 建议把 num_processes 设为 1~2
    devices = ["cuda:0", "cuda:1"]
    
    with Pool(
        processes=args.num_processes,
        initializer=init_worker,
        initargs=(args.input_dir, args.output_dir, args.device)
    ) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_video, all_files), total=len(all_files)))

    # 打印失败项（可选）
    failed = [p for status, p in results if status != "ok"]
    if failed:
        print(f"[WARN] Failed videos: {len(failed)}")
        for p in failed[:20]:
            print("  ", p)

if __name__ == "__main__":
    main()

# # Create a RetinaFace detector using Resnet50 backbone, with the confidence
# # threshold set to 0.8
# face_detector = RetinaFacePredictor(
#     threshold=0.8, device='cuda:0',
#     model=RetinaFacePredictor.get_model('mobilenet0.25'))

# # Create a facial landmark detector
# landmark_detector = FANPredictor(
#     device='cuda:0', model=FANPredictor.get_model('2dfan2_alt'))

# # ipdb.set_trace()

# for root, file_name in tqdm(all_files):
#     input_path = os.path.join(root, file_name)
#     rel_path = os.path.relpath(input_path, args.input_dir)
#     output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '.npy')
#     vis_path = os.path.join(args.vis_dir, rel_path) if args.vis_dir else None

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)


#     cap = cv2.VideoCapture(os.path.join(root, file_name))
#     if not cap.isOpened():
#         print(f"[WARN] Cannot open video: {os.path.join(root, file_name)}")
#         continue

#     all_landmarks = []

#     while True:
#         ret, image = cap.read()   # image is a frame (BGR)
#         if not ret:
#             break

#         detected_faces = face_detector(image, rgb=False)
#         landmarks, scores = landmark_detector(image, detected_faces, rgb=False)

#         # 每帧保存一次（landmarks 可能是 list/ndarray，直接 append）
#         all_landmarks.append(landmarks)

#     cap.release()

#     np.save(output_path, np.array(all_landmarks, dtype=object), allow_pickle=True)


        