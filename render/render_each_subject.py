import json
import os


def load_dataset(dataset_json):
    ret=list()
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    for subject in dataset:
        for emo in dataset[subject]:
            if emo != 'neutral':
                continue
            for level in dataset[subject][emo]:
                for vid in dataset[subject][emo][level]:
                    ret.append([
                        dataset[subject][emo][level][vid]["param"],
                        dataset[subject][emo][level][vid]["video"]
                                ])
                    break

    return ret



if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    raw_json = '/data/chenziang/codes/smirk/index.json'
    device = 'cuda'
    params = load_dataset(raw_json)

    for param, vid in params:
        # 获取param上一级目录
        _param = '/'.join(param.split('/')[:-1])
        cmd = f"python render/render_mead.py \
            --param_path {_param} \
            --vid_path {vid} \
            --out_path output/ "
        os.system(cmd)

