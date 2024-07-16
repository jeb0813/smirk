import os 
import cv2
import json
import librosa

from scipy.io.wavfile import read

import warnings
warnings.filterwarnings('ignore')


def remove_extra_cam_views(dataset_path):
    for subject in sorted(os.listdir(dataset_path)):
        video_path = os.path.join(dataset_path, subject, 'video')

        for view in sorted(os.listdir(video_path)):
            if view != 'front':
                view_path = os.path.join(video_path, view)
                # print(view_path)
                print(f'Removing {view} from {subject}...') 
                os.system(f'rm -r {view_path}')

def remove_view_folder(dataset_path):
    for subject in sorted(os.listdir(dataset_path)):
        video_path = os.path.join(dataset_path, subject, 'video')

        # 检查是否只存在 front 视角
        # assert len(os.listdir(video_path)) == 1 and os.listdir(video_path)[0] == 'front'

        print(f'Moving front view to {subject}...')
        front_view_path = os.path.join(video_path, 'front')

        if not os.path.exists(front_view_path):
            print(f'No front view found in {subject}')
            continue
        # print(f'mv {front_view_path}/* {video_path}')
        os.system(f'mv {front_view_path}/* {video_path}')
        os.system(f'rm -r {front_view_path}')
        # exit()


def check_vid_num(dataset_path):

    for subject in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject)

        # 检查是否只存在 video 和 audio 文件夹
        assert len(os.listdir(subject_path)) == 2 and 'video' in os.listdir(subject_path) and 'audio' in os.listdir(subject_path), f'{subject}'

        d_audio, d_video = get_subject_struct(subject_path)
        print(d_audio)
        exit()
        
        # 检查 audio 和 video 长度是否一致
        for emo in d_audio:
            for level in d_audio[emo]:
                assert len(d_audio[emo][level]) == len(d_video[emo][level]), f'{subject} {emo} {level} {len(d_audio[emo][level])} {len(d_video[emo][level])}'
                print(f'{subject} {emo} {level} {len(d_audio[emo][level])} {len(d_video[emo][level])}')



def get_subject_struct(subject_path):
    d_audio = dict()
    d_video = dict()

    # 先audio
    audio_path = os.path.join(subject_path, 'audio')
    for emo in sorted(os.listdir(audio_path)):
        d_audio[emo] = dict()
        
        emo_path = os.path.join(audio_path, emo)
        for level in sorted(os.listdir(emo_path)):
            d_audio[emo][level] = dict()
            level_path = os.path.join(emo_path, level)
            
            for vid in sorted(os.listdir(level_path)):
                d_audio[emo][level][vid] = os.path.join(level_path, vid)



    # 再video
    video_path = os.path.join(subject_path, 'video')
    for emo in sorted(os.listdir(video_path)):
        d_video[emo] = dict()
        
        emo_path = os.path.join(video_path, emo)
        for level in sorted(os.listdir(emo_path)):
            d_video[emo][level] = dict()
            level_path = os.path.join(emo_path, level)
            
            for vid in sorted(os.listdir(level_path)):
                d_video[emo][level][vid] = os.path.join(level_path, vid)
    
    return d_audio, d_video


def rename_vids(dataset_path):
    for subject in sorted(os.listdir(dataset_path)):
        # 过滤已经处理过的
        if subject[0]=='M' or (subject[0]=='W' and int(subject[1:]) < 14):
            continue

        subject_path = os.path.join(dataset_path, subject, 'video')
        for emo in sorted(os.listdir(subject_path)):
            emo_path = os.path.join(subject_path, emo)
            for level in sorted(os.listdir(emo_path)):
                level_path = os.path.join(emo_path, level)
                vids = sorted(os.listdir(level_path))

                for i,vid in enumerate(vids):
                    correct_name = f"{i+1:03d}.mp4"
                    print(subject,emo,level,vid)

                    aud_path = os.path.join(dataset_path, subject, 'audio', emo, level, correct_name.replace('mp4','m4a'))
                    vid_path = os.path.join(level_path, vid)

                    # 名称正确可以容忍长度不对，直接trim
                    if vid == correct_name:
                        if not check_vid_aud_pair(aud_path, vid_path):
                            audio, sr = librosa.load(path=aud_path, sr=None)
                            aud_duration = librosa.get_duration(y=audio, sr=sr)

                            cap = cv2.VideoCapture(vid_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            vid_duration = total_frames / fps

                            if aud_duration > vid_duration:
                                print('audio longer than video')
                                print('audio:', aud_duration)
                                print('video:', vid_duration)
                                print('trimming audio...')
                                os.system(f'ffmpeg -y -i {aud_path} -acodec copy -t {vid_duration} -c copy {aud_path}')
                            else:
                                print('video longer than audio')
                                print('audio:', aud_duration)
                                print('video:', vid_duration)
                                print('trimming video...')
                                os.system(f'ffmpeg -y -i {vid_path} -vcodec copy -t {aud_duration} -c copy {vid_path}')
                    else:
                        print('wrong name!!!')
                        print('curr name:', vid)
                        print('correct name:', correct_name)

                        # 名称不对时间对的话，直接改名
                        if not check_vid_aud_pair(aud_path, vid_path):
                            raise ValueError(f'{subject} {emo} {level} {vid}')
                        else:
                            print('correcting name...')
                            os.system(f'mv {os.path.join(level_path, vid)} {os.path.join(level_path, correct_name)}')

                    # assert check_vid_aud_pair( \
                    #     os.path.join(dataset_path, subject, 'audio', emo, level, correct_name.replace('mp4','m4a')), \
                    #         os.path.join(level_path, vid)\
                    #             ), f'{subject} {emo} {level} {vid}'

                    


def check_vid_aud_pair(aud_path,vid_path):
    # 读取音频文件
    audio, sr = librosa.load(path=aud_path, sr=None)
    aud_duration = librosa.get_duration(y=audio, sr=sr)
    
    # 获取视频
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    vid_duration = total_frames / fps

    print('audio:', aud_duration)
    print('video:', vid_duration)

    # 误差小于0.1s即合法
    return abs(aud_duration - vid_duration) <= 0.1




if __name__ == '__main__':
    dataset_path = '/data/chenziang/codes/smirk/MEAD'
    # remove_extra_cam_views(dataset_path)
    # remove_view_folder(dataset_path)
    # check_vid_num(dataset_path)
    rename_vids(dataset_path)

    # ret = check_vid_aud_pair('/data/chenziang/codes/smirk/MEAD/M030/audio/fear/level_3/027.m4a',\
    #                    '/data/chenziang/codes/smirk/MEAD/M030/video/fear/level_3/027.mp4'
    #                 )
    # print(ret)
    pass
    

