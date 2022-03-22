import glob
import json
import os
import ndjson
import random

def file2paralist(file_path):
      global para_sum
      with open(file_path, 'r') as f:
        query_paras = f.readlines()
      start = False
      para_list = []
      paragraph = []
      for line in query_paras:
        if not start:
          if '[1]' in line:
            start = True
            i = 2
          continue
        if not f'[{i}]' in line:
          paragraph.append(line)
        else:
          paragraph = ''.join(paragraph)
          
          paragraph = paragraph.replace('\n ', '').replace('\n', '')
          para_list.append(paragraph)
          paragraph = []
          i += 1
      para_sum += i
      return para_list
      

if __name__ == "__main__":

  random.seed(0)
  para_sum = 0
  task1_label_path = './data/task1/task1_train_labels_2022.json'
  task1_files_path = './data/task1/task1_train_files'
  task1_case_para = './data/task1/task1_train_files/train_data.json'

  task1_train_data_path = './data/task1/train_data_50.json'
  task1_val_data_path = './data/task1/val_data_50.json'
  task1_test_data_path = './data/task1/test_data_50.json'

  save_paths = [task1_train_data_path, task1_val_data_path, task1_test_data_path]

  candidate_file_paths = glob.glob(os.path.join(task1_files_path, '*'))

  with open(task1_label_path, 'r') as f:
    task1_label_dict = json.load(f)



  c_paras_list = {}
  train, val, test, support = 0, 0, 0, 0
  for path in save_paths:
      if os.path.isfile(path):
          os.remove(path)
  for i, (query_txt, docs_txt) in enumerate(task1_label_dict.items(), 1):
      print(f'{i}/{len(task1_label_dict)}')
      value = random.random()
      if value < 0.6:
          path = save_paths[0]  
          train += 1
      elif value < 0.8:
          path = save_paths[1]
          val += 1
      else:
          path = save_paths[2]
          test += 1
      query_path = os.path.join(task1_files_path, query_txt)
      docs_txt_paths = [os.path.join(task1_files_path, d) for d in docs_txt]
      support += len(docs_txt_paths)
      tmp_candidate_file_paths = random.sample(candidate_file_paths, 50 - len(docs_txt_paths)) + docs_txt_paths
      tmp_candidate_file_paths = list(set(tmp_candidate_file_paths))
      query = query_txt.replace('.txt', '')
      q_paras = file2paralist(query_path)
      data = {}
      data['q_paras'] = q_paras
      string = ''
      for j, candidate_file_path in enumerate(tmp_candidate_file_paths):
          candidate_txt = candidate_file_path.split('/')[-1]
          candidate = candidate_txt.replace('.txt', '')
          data['guid'] = query + '_' + candidate
          if candidate_txt in docs_txt:
            label = 1
          else:
            label = 0
          c_paras = file2paralist(candidate_file_path) if not candidate_file_path in c_paras_list else c_paras_list[candidate_file_path]
          #if not candidate_file_path in c_paras_list:
          #  c_paras_list[candidate_file_path] = c_paras
          data['c_paras'] = c_paras
          data['label'] = label
          string += (json.dumps(data) + '\n')
      with open(path, 'a') as f:
          f.write(string)
      if i == 450:
            break

  print(train, val, test, support/450)