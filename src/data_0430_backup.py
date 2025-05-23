# src/data.py

from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer
from tokenization import P5Tokenizer

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)   

image_feature_dim_dict = {
    'vitb32': 512,
    'vitb16': 512,
    'vitl14': 768,
    'rn50': 1024,
    'rn101': 512
}

class VIP5_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', 
                 data_root='data',         # <--- 新增
                 feature_root='features',  # <--- 新增
                 sample_type='random'):
        
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.sample_type = sample_type
        self.image_feature_size_ratio = args.image_feature_size_ratio  # such as 1, 2, 3, 5, 10
        self.image_feature_type = args.image_feature_type
        assert self.image_feature_type in ['vitb32', 'vitb16', 'vitl14', 'rn50', 'rn101']
        self.image_feature_dim = image_feature_dim_dict[self.image_feature_type]
        self.feature_root = feature_root   # <--- 新增
        self.data_root = data_root         # <--- 新增
        

        print('Data sources: ', split.split(','))
        self.mode = mode

        # --- 规范化 attack_mode，决定是否使用投毒数据 --- #
        # 不要用“mode”以免和 self.mode 冲突
        attack_mode = getattr(self.args, "attack_mode", "none").lower()
        use_poison = attack_mode not in ("none", "noattack")

  
        # —— 根据 use_poison 选择加载原始或投毒后的 exp_splits —— #
        exp_fname = "exp_splits_poisoned.pkl" if use_poison else "exp_splits.pkl"
        exp_splits = load_pickle(os.path.join(self.data_root, self.split, exp_fname))
        if self.mode == 'train':
            self.exp_data = exp_splits['train']
        elif self.mode == 'val':
            self.exp_data = exp_splits['val']
        elif self.mode == 'test':
            self.exp_data = exp_splits['test']
        else:
            raise NotImplementedError
           
            
        # —— DEBUG: 打印 exp_fname 和前 5 条 exp_data —— #
        print(f"[DEBUG] exp_fname={exp_fname}")
        print("[DEBUG] exp_data[:5]:", self.exp_data[:5])


        # —— 根据 use_poison 选择加载原始或投毒后的 sequential_data —— #
        seq_fname = "sequential_data_poisoned.txt" if use_poison else "sequential_data.txt"
        self.sequential_data = ReadLineFromFile(os.path.join(self.data_root, self.split, seq_fname))


        # —— DEBUG: 打印前 5 条，确认切换是否生效 —— #
        print(f"[DEBUG] attack_mode={attack_mode}, use_poison={use_poison}, seq_fname={seq_fname}")
        print("[DEBUG] sequential_data[:5]:", self.sequential_data[:5])
        print("[DEBUG] exp_data[:5]:", self.exp_data[:5])


        item_count = defaultdict(int)
        user_items = defaultdict()
        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum(count)
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join(self.data_root, self.split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join(self.data_root, self.split, 'datamaps.json'))
        # 加载完 datamaps 之后，统一把 user2id 的所有 key 都转成 str
        self.user2id = { str(k): v for k, v in datamaps['user2id'].items() }
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        
        
        # 根据 use_poison 决定加载哪份映射：use_poison 时优先 poisoned，否则强制原始
        poisoned_map = os.path.join(self.data_root, self.split, 'user_id2name_poisoned.pkl')
        orig_map     = os.path.join(self.data_root, self.split, 'user_id2name.pkl')

        if not use_poison:
            if os.path.exists(orig_map):
                print(f"[INFO] (NoAttack) 加载原始映射: {orig_map}")
                self.user_id2name = load_pickle(orig_map)
            else:
                raise FileNotFoundError(f"找不到原始映射文件: {orig_map}")
        else:
            if os.path.exists(poisoned_map):
                print(f"[INFO] 加载扩展映射: {poisoned_map}")
                self.user_id2name = load_pickle(poisoned_map)
            elif os.path.exists(orig_map):
                print(f"[INFO] 加载原始映射: {orig_map}")
                self.user_id2name = load_pickle(orig_map)
            else:
                raise FileNotFoundError(
                    f"未在 {os.path.join(self.data_root, self.split)} 找到 user_id2name.pkl 或 user_id2name_poisoned.pkl"
                )


        # 根据映射中第一个键的类型确定转换函数
        if self.user_id2name:
            first_key = next(iter(self.user_id2name))
            self.key_convert = int if isinstance(first_key, int) else str
        else:
            self.key_convert = str

        self.meta_data = []
        for meta in parse(os.path.join(self.data_root, self.split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i
            
        # Visual features
        self.item2img_dict = load_pickle(os.path.join(self.data_root, self.split, 'item2img_dict.pkl'))
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'sequential':
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 9 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([6 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'direct':
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            else:
                raise NotImplementedError
    
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args
        loss_weight = 1.0
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            # 使用转换函数将读取的 user_id 转换为映射中键的类型
            uid = self.key_convert(user_id)
            if uid not in self.user_id2name:
                print(f"[WARN] 用户ID {uid} 不在映射中，使用默认 placeholder")
                user_desc = f"synthetic_user_{uid}"
            else:
                user_desc = self.user_id2name[uid]




            if self.mode == 'train':
                # 修改点1：生成 end_candidates
                end_candidates = list(range(
                    max(2, len(sequence) - 6),
                    len(sequence) - 3
                ))

                # 修改点2：fallback 并打印 datum_idx（而非 idx）
                if not end_candidates:
                    print(f"[ERROR] datum_idx={datum_idx} 无合法 end_candidates，"
                          f"sequence={sequence!r}")
                    # fallback：取倒数第二个位置保证不越界
                    end_candidates = [len(sequence) - 2]

                # 修改点3：直接选出 end_pos
                end_pos = random.choice(end_candidates)



                # 修改点4：同样为 start_candidates 增加 fallback
                start_candidates = list(range(1, min(4, end_pos)))
                if not start_candidates:
                    print(f"[ERROR] datum_idx={datum_idx} 无合法 start_candidates，"
                          f"end_pos={end_pos}")
                    start_candidates = [1]
                start_pos = random.choice(start_candidates)


                purchase_history = sequence[start_pos : end_pos + 1]
                target_item     = sequence[end_pos + 1]







            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] in ['A-1', 'A-2', 'A-3']:
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                else:
                    source_text = task_template['source'].format(user_id, ' {}-> '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(purchase_history), self.image_feature_dim), dtype=np.float32)
                for i in range(len(purchase_history)):
                    feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[purchase_history[i]] + '.npy'))
            # 以下部分保持原逻辑，未做修改……
            elif task_template['id'] in ['A-4', 'A-5', 'A-6', 'A-9']:
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                else:
                    source_text = task_template['source'].format(user_desc, ' {}-> '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(purchase_history), self.image_feature_dim), dtype=np.float32)
                for i in range(len(purchase_history)):
                    feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[purchase_history[i]] + '.npy'))
            elif task_template['id'] == 'A-7':
                symbol_prob = random.random()
                symbol = ' {}, ' if symbol_prob > 0.5 else ' {}-> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[purchase_history[i]] + '.npy'))
                    feats[-1] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[target_item] + '.npy'))
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                      if self.sample_type == 'random'
                                      else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, candidate_samples[0], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[purchase_history[i]] + '.npy'))
                    feats[-1] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[0]] + '.npy'))
            elif task_template['id'] == 'A-8':
                symbol_prob = random.random()
                symbol = ' {}, ' if symbol_prob > 0.5 else ' {}-> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[purchase_history[i]] + '.npy'))
                    feats[-1] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[target_item] + '.npy'))
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                      if self.sample_type == 'random'
                                      else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, candidate_samples[0], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[purchase_history[i]] + '.npy'))
                    feats[-1] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[0]] + '.npy'))
            else:
                raise NotImplementedError
                
        elif task_name == 'direct':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            uid = self.key_convert(user_id)
            if uid not in self.user_id2name:
                print(f"[WARN] 用户ID {uid} 不在映射中，使用默认 placeholder")
                user_desc = f"synthetic_user_{uid}"
            else:
                user_desc = self.user_id2name[uid]
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates) - 1)
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)
            task_template = self.all_tasks['direct'][task_candidates[task_idx]]
            assert task_template['task'] == 'direct'
            if task_template['id'] == 'B-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[target_item] + '.npy'))
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                      if self.sample_type == 'random'
                                      else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[0]] + '.npy'))
            elif task_template['id'] == 'B-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>', user_desc)
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[target_item] + '.npy'))
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                      if self.sample_type == 'random'
                                      else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(candidate_samples[0], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>', user_desc)
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[0]] + '.npy'))
            elif task_template['id'] == 'B-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[target_item] + '.npy'))
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                      if self.sample_type == 'random'
                                      else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[0]] + '.npy'))
            elif task_template['id'] == 'B-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[target_item] + '.npy'))
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                      if self.sample_type == 'random'
                                      else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[0]] + '.npy'))
            elif task_template['id'] in ['B-5', 'B-6']:
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                  if self.sample_type == 'random'
                                  else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(candidate_samples) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(candidate_samples), self.image_feature_dim), dtype=np.float32)
                for i in range(len(candidate_samples)):
                    feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[i]] + '.npy'))
            elif task_template['id'] in ['B-7', 'B-8']:
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    sample_ids = (np.random.choice(self.all_item, candidate_num, replace=False)
                                  if self.sample_type == 'random'
                                  else np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability))
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_id, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(candidate_samples) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(candidate_samples), self.image_feature_dim), dtype=np.float32)
                for i in range(len(candidate_samples)):
                    feats[i] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, self.id2item[candidate_samples[i]] + '.npy'))
            else:
                raise NotImplementedError
                
        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            if task_template['id'] == 'C-1':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-2':
                source_text = task_template['source'].format(exp_datum['summary'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-3':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-4':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-5':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['summary'], user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-6':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, int(exp_datum['overall']), self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-7':
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-8':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-9':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-10':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-11':
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            elif task_template['id'] == 'C-12':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = np.load(os.path.join(self.feature_root, f'{self.image_feature_type}_features', self.split, exp_datum['asin'] + '.npy'))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        category_ids = [1 if token_id == 32099 else 0 for token_id in input_ids]  # 32099 为 '<extra_id_0>' 的 token id
        
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)
        
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['category_ids'] = torch.LongTensor(category_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        
        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text
        out_dict['task'] = task_template['task']
        
        feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats
        out_dict['vis_feat_length'] = feats.shape[0]
        out_dict['loss_weight'] = loss_weight
        
        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁') or tokenized_text[i] == '<extra_id_0>':
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        # 添加一个 0 表示 </s>
        return whole_word_ids[:len(input_ids) - 1] + [0]
    
    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)
        args = self.args
        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)
        V_W_L = max(entry['vis_feat_length'] for entry in batch)
        
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        category_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        vis_feats = torch.zeros(B, V_W_L, self.image_feature_dim)
        loss_weights = torch.ones(B, dtype=torch.float)
        
        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            category_ids[i, :entry['input_length']] = entry['category_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            vis_feats[i, :entry['vis_feat_length']] = entry['vis_feats']
            if 'task' in entry:
                tasks.append(entry['task'])
            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])
            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight'] / entry['target_length'] if entry['target_length'] > 0 else entry['loss_weight']
        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks
        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text
        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['category_ids'] = category_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['vis_feats'] = vis_feats
        batch_entry['loss_weights'] = loss_weights
        
        return batch_entry
    
def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
               batch_size=16, workers=4, distributed=False, 
               data_root='data',        # <--- 新增
               feature_root='features'  # <--- 新增
               ):
    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case)
    
    from all_templates import all_tasks as task_templates
    dataset = VIP5_Dataset(
        task_templates,
        task_list,
        tokenizer,
        args,
        sample_numbers,
        mode=mode,
        split=split,
        data_root=data_root,
        feature_root=feature_root
    )
    
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader
