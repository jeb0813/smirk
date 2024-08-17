import json



if __name__ == "__main__":
    dataset_json_path = '/data/chenziang/codes/smirk/index.json'
    save_path = '/data/chenziang/codes/smirk/ids.json'
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    subjects = list(dataset.keys())
    subjects.sort()

    ids = {k:i+1 for i,k in enumerate(subjects)}

    with open(save_path, 'w') as f:
        json.dump(ids, f, indent=4)

    