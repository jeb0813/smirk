import os
import numpy as np

def get_templates(dataset_path):
    """
    Get the templates from the dataset
    :param dataset_path: path to the dataset
    :return: templates
    """
    templates = {}

    for subject in sorted(os.listdir(dataset_path)):
        param_path = os.path.join(dataset_path, subject, 'param', 'neutral', 'level_1', '001', '001.npz')

        if not os.path.exists(param_path):
            raise FileNotFoundError(f'param_path not found: {param_path}')
        
        param = np.load(param_path)

        # 将数据从 NpzFile 对象提取到一个可变的字典中
        param_dict = {key: param[key] for key in param}
        
        # set other params to ZERO
        remove_keys = ('pose_params', 'eyelid_params')
        for key in remove_keys:
            param_dict[key] = np.zeros_like(param_dict[key])

        templates[subject] = param_dict
    
    return templates


if __name__ == "__main__":
    dataset_path = 'MEAD'
    save_path = 'templates'
    templates = get_templates(dataset_path)
    for key, value in templates.items():
        print(key)
        np.savez(os.path.join(save_path, f'{key}_no_eyelid.npz'), **value)
        # np.savez(save_filename, **templates)

        