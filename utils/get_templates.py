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
        templates[subject] = param
    
    return templates


if __name__ == "__main__":
    dataset_path = 'MEAD'
    save_path = 'templates'
    templates = get_templates(dataset_path)
    for key, value in templates.items():
        print(key)
        np.savez(os.path.join(save_path, f'{key}.npz'), **value)
        # np.savez(save_filename, **templates)

