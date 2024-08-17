import os
import numpy as np
import cv2
import librosa
import json

def get_index(dataset_path):
    d = dict()
    summary = dict()

    for subject in sorted(os.listdir(dataset_path)):
        # if subject[0] == 'M' and int(subject[1:]) < 30:
        #     continue

        cnt_subject = 0
        subject_dict = dict()
        subject_summary_dict = dict()

        d[subject] = subject_dict
        summary[subject] = subject_summary_dict

        subject_path = os.path.join(dataset_path, subject, 'video')

        for emo in sorted(os.listdir(subject_path)):
            cnt_emo = 0
            emo_dict = dict()
            emo_summary_dict = dict()

            subject_dict[emo] = emo_dict
            subject_summary_dict[emo] = emo_summary_dict

            emo_path = os.path.join(subject_path, emo)

            for level in sorted(os.listdir(emo_path)):
                # cnt_level = 0
                level_dict = dict()
                # level_summary_dict = dict()

                emo_dict[level] = level_dict
                # emo_summary_dict[level] = level_summary_dict

                level_path = os.path.join(emo_path, level)

                for vid in sorted(os.listdir(level_path)):
                    vid_dict = dict()
                    vid_summary_dict = dict()

                    level_dict[vid] = vid_dict
                    # level_summary_dict[vid] = vid_summary_dict

                    vid_path = os.path.join(level_path, vid)
                    audio_path = vid_path.replace('video', 'audio').replace('mp4', 'm4a')
                    param_path = os.path.join(vid_path.replace('video', 'param').replace('.mp4', ''), 'param.npz')

                    print(f'Processing {vid_path}...')

                    if not os.path.exists(audio_path):
                        print(f'{audio_path} not exists')
                        continue
                    
                    if not os.path.exists(param_path):
                        print(f'{param_path} not exists')
                        continue
                    # assert os.path.exists(audio_path), f'{audio_path} not exists'
                    # assert os.path.exists(param_path), f'{param_path} not exists'

                    # 检查长度
                    cap = cv2.VideoCapture(vid_path)
                    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    param_length = np.load(param_path)['cam'].shape[0]

                    if vid_length != param_length:
                        print(f'{vid_path} and {param_path} have different lengths {vid_length} vs {param_length}')
                        continue
                    # assert vid_length == param_length, f'{vid_path} and {param_path} have different lengths {vid_length} vs {param_length}'
                    vid_summary_dict['length'] = vid_length

                    vid_dict['video'] = vid_path
                    vid_dict['audio'] = audio_path
                    vid_dict['param'] = param_path

                    # cnt_level += 1
                    cnt_emo += 1
                    cnt_subject += 1
                
                # level_summary_dict['count'] = cnt_level
            emo_summary_dict['count'] = cnt_emo
        subject_summary_dict['count'] = cnt_subject

        # break

    return d, summary


if __name__ =="__main__":
    dataset_path = '/data/chenziang/codes/smirk/MEAD'

    d, summary = get_index(dataset_path)

    with open('index.json', 'w') as f:
        json.dump(d, f, indent=4)
    
    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

