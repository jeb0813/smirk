import os
import json



if __name__ == "__main__":
    raw_json = '/data/chenziang/codes/smirk/index.json'
    test_num = 5

    with open(raw_json, 'r') as f:
        data = json.load(f)

    train = dict()
    test = dict()

    for subject in data:
        train[subject] = dict()
        test[subject] = dict()

        for emo in data[subject]:
            train[subject][emo] = dict()
            test[subject][emo] = dict()

            for level in data[subject][emo]:
                all_vids = data[subject][emo][level]
                all_keys = list(data[subject][emo][level].keys())
                
                train_vids = {k:all_vids[k] for k in all_keys[:-test_num]}
                test_vids = {k:all_vids[k] for k in all_keys[-test_num:]}

                train[subject][emo][level] = train_vids
                test[subject][emo][level] = test_vids

    with open('/data/chenziang/codes/smirk/train.json', 'w') as f:
        json.dump(train, f, indent=4)
    
    with open('/data/chenziang/codes/smirk/test.json', 'w') as f:
        json.dump(test, f, indent=4)

    
    




