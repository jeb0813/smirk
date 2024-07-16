import numpy as np 
import os
import re

def combine_npz(dataset_path):
    for subject in sorted(os.listdir(dataset_path)):
        subject_param_path = os.path.join(dataset_path, subject, 'param')
        for emo in sorted(os.listdir(subject_param_path)):
            emo_param_path = os.path.join(subject_param_path, emo)
            for level in sorted(os.listdir(emo_param_path)):
                level_param_path = os.path.join(emo_param_path, level)
                for vid in sorted(os.listdir(level_param_path)):
                    param_path = os.path.join(level_param_path, vid)
                    print(f'Processing {param_path}...')
                    # 正则过滤文件名, <三位数字>.npz
                    pattern = re.compile(r'^\d{3}\.npz$')
                    params = sorted([f for f in os.listdir(param_path) if pattern.match(f)])

                    first_file = np.load(os.path.join(param_path, params[0]))
                    fields = first_file.files

                    # 初始化一个字典用于存储合并后的数据
                    merged_data = {}
                    for field in fields:
                        first_array = first_file[field]
                        # 创建一个新的数组，预分配足够的空间
                        merged_data[field] = np.empty((len(params),) + first_array.shape, dtype=first_array.dtype)
                        merged_data[field][0] = first_array
                    
                    # 读取其余文件的数据并将其存储到合并后的数组中
                    for i, file in enumerate(params[1:], start=1):
                        npz_data = np.load(os.path.join(param_path, file))
                        for field in fields:
                            merged_data[field][i] = npz_data[field]

                    # 保存
                    np.savez(os.path.join(param_path, 'param.npz'), **merged_data)

if __name__ == "__main__":
    dataset_path = 'MEAD'

    combine_npz(dataset_path)


