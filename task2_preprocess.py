import glob
import json
import os
import re
import ndjson
import random

if __name__ == "__main__":
    random.seed(0)

    with open('./data/task2/task2_train/task2_train_labels_2022.json', 'r') as f:
        train_labels = json.load(f)

    task2_raw_path = './data/task2/task2_train/'
    task2_train_data_path = './data/task2/train_data_225.json'
    task2_val_data_path = './data/task2/val_data_225.json'
    task2_test_data_path = './data/task2/test_data_225.json'

    save_paths = [task2_train_data_path, task2_val_data_path, task2_test_data_path]

    for path in save_paths:
        if os.path.isfile(path):
            os.remove(path)
    j=0
    train, val, test, support = 0, 0, 0, 0
    for i, (folder_num, label_list) in enumerate(train_labels.items()):
        print(f'{i}/{len(train_labels)}')
        label_list = label_list[0].split(', ')
        support += len(label_list)
        value = random.random()
        if value < 0.8:
            path = save_paths[0]
            train += 1
        elif value < 1:
            path = save_paths[1]
            val += 1
        else:
            path = save_paths[2]
            test += 1

        folder_path = os.path.join(task2_raw_path, folder_num)
        data = {}
        with open(os.path.join(folder_path, 'entailed_fragment.txt'), 'r') as f:
            text_a = f.read()
        data['text_a'] = text_a

        para_paths = glob.glob(os.path.join(folder_path, 'paragraphs/*'))

        for para_path in para_paths:
            text_name, text_num = re.search(r'(\d{3}).txt', para_path).group(), re.search(r'(\d{3}).txt', para_path).group(1)
            guid = f'{folder_num}_{text_num}'
            data['guid'] = guid
            with open(para_path, 'r') as f:
                text_b = f.readlines()
                text_b = ''.join(text_b[1:])
            data['text_b'] = text_b
            if text_name in label_list:
                label = 1
            else:
                label = 0
            data['label'] = label
            with open(path, 'a') as f:
                writer = ndjson.writer(f)
                writer.writerow(data)
        j += len(para_paths)
        if i == 224:
            break
    print(j / 225, train, val, test, support / 225)
