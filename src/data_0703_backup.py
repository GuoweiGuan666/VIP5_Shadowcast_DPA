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
import yaml
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
    def __init__(
        self,
        all_tasks,
        task_list,
        tokenizer,
        args,
        sample_numbers,
        mode='train',
        split='toys',
        data_root='data',
        feature_root='features',
        sample_type='random'  
    ):
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.sample_type = sample_type
        self.image_feature_size_ratio = args.image_feature_size_ratio
        self.image_feature_type = args.image_feature_type
        assert self.image_feature_type in ['vitb32', 'vitb16', 'vitl14', 'rn50', 'rn101']
        self.image_feature_dim = image_feature_dim_dict[self.image_feature_type]
        self.feature_root = feature_root
        self.data_root = data_root
        self.mode = mode

        # 1) 直接用命令行传入的 args.attack_mode 和 args.mr
        atk = self.args.attack_mode       # e.g. "RandomInjectionAttack" / "NoAttack" / ...
        mr  = self.args.mr                # e.g. 0.1, 0.2, etc.



        # CamelCase -> snake_case，再去掉末尾 _attack
        import re
        def camel_to_snake(name: str) -> str:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        atk_snake = camel_to_snake(atk).replace("_attack", "")
        # —— 把 "no" 也当作 NoAttack
        if atk_snake == "no":
            atk_snake = "noattack"

        # —— alias 映射：脚本里用 DirectBoostingAttack，但文件夹里是 direct_boost
        # ↓ 加入别名映射
        if atk_snake == "direct_boosting":
            atk_snake = "direct_boost"

        # PopularItemMimickingAttack  →  popular_mimicking
        if atk_snake == "popular_item_mimicking":
            atk_snake = "popular_mimicking"


        # 数字 MR 转字符串
        # e.g. 0.1 -> "0.1", 0.2 -> "02"
        mr_str = str(int(mr)) if float(mr).is_integer() else str(mr)

        if atk_snake not in ("none", "noattack"):
            # 有毒文件都在 data/<split>/poisoned 下
            pois = os.path.join(self.data_root, self.split, "poisoned")
            exp_splits_path = os.path.join(pois, f"exp_splits_{atk_snake}_mr{mr_str}.pkl")
            seq_path        = os.path.join(pois, f"sequential_data_{atk_snake}_mr{mr_str}.txt")
            idx_path        = os.path.join(pois, f"user_id2idx_{atk_snake}_mr{mr_str}.pkl")
            name_path       = os.path.join(pois, f"user_id2name_{atk_snake}_mr{mr_str}.pkl")
        else:
            # NoAttack ：读取原始数据
            base = os.path.join(self.data_root, self.split)
            exp_splits_path = os.path.join(base, "exp_splits.pkl")
            seq_path        = os.path.join(base, "sequential_data.txt")
            idx_path        = os.path.join(base, "user_id2idx.pkl")
            name_path       = os.path.join(base, "user_id2name.pkl")



        # —— DEBUG 打印，确认到底加载的是哪个文件
        print(f"[DEBUG] exp_splits_path = {exp_splits_path}")
        print(f"[DEBUG] seq_path        = {seq_path}")
        print(f"[DEBUG] idx_path        = {idx_path}")
        print(f"[DEBUG] name_path       = {name_path}")

        # 加载 split / seq / user 映射
        exp_splits = load_pickle(exp_splits_path)

        if self.mode == 'train':
            self.exp_data = exp_splits['train']
        elif self.mode == 'val':
            self.exp_data = exp_splits['val']
        elif self.mode == 'test':
            self.exp_data = exp_splits['test']
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")


        # 3) 加载 sequential_data 文件
        #    （路径已经在上面根据攻击模式和 mr 算好了）
        self.sequential_data = ReadLineFromFile(seq_path)

        # —— 只在 poisoned 且 val/test 模式下过滤掉新注入的 fake 用户
        if atk_snake not in ("none", "noattack") and self.mode in ("val", "test"):
            # 1) 先一次性读原始 un-poisoned 序列，拿合法用户集合
            orig_path = os.path.join(self.data_root, self.split, "sequential_data.txt")
            orig_users = { line.split()[0] for line in ReadLineFromFile(orig_path) }
            # 2) 用列表推导保留那些合法的 sequential_data
            before_seq = len(ReadLineFromFile(seq_path))
            self.sequential_data = [
                line for line in self.sequential_data
                if line.split()[0] in orig_users
            ]

            print(f"[DEBUG] Val/Test 模式下，剔除了 {before_seq - len(self.sequential_data)} 条 fake 用户数据")

            # —— **不要** 再去过滤 self.exp_data ！解释任务需要保留所有 review 样本
    

        # 4) 构建 user_items & 统计 item_count 用于采样
        item_count = defaultdict(int)
        user_items = {}
        for line in self.sequential_data:
            user, items_str = line.strip().split(' ', 1)
            items = [int(x) for x in items_str.split()]
            user_items[user] = items
            for it in items:
                item_count[it] += 1
        self.all_item = list(item_count.keys())
        counts = np.array(list(item_count.values()), dtype=float)
        self.probability = (counts / counts.sum()).tolist()
        self.user_items = user_items

        # 如果是 test 模式，加载 negative_samples.txt
        if self.mode == 'test':
            neg_path = os.path.join(self.data_root, self.split, 'negative_samples.txt')
            self.negative_samples = ReadLineFromFile(neg_path)


        # 5) 加载 user_id2idx/user_id2name 映射
        #    （路径已经在上面根据攻击模式和 mr 算好了）
        # 在只有 Explanation 任务时，直接动态构建 reviewerID 的映射
        if set(self.task_list.keys()) == {"explanation"}:
            raw_user2id = {}
            self.user_id2name = {}
            for exp in self.exp_data:
                uid = exp.get("reviewerID")
                if uid not in raw_user2id:
                    raw_user2id[uid] = len(raw_user2id)
                    # 如果有 reviewerName，就用它，否则用 ID
                    self.user_id2name[uid] = exp.get("reviewerName", uid)
            print(f"[DEBUG] Explanation-only，动态构建了 {len(raw_user2id)} 个用户映射")
        else:
            # 否则按原逻辑，从文件里读
            if not os.path.exists(idx_path) or not os.path.exists(name_path):
                if atk_snake in ("none", "noattack"):
                    # NoAttack 再 fallback 动态构建（跟上面类似）
                    raw_user2id = {}
                    self.user_id2name = {}
                    for line in self.sequential_data:
                        uid = line.split()[0]
                        if uid not in raw_user2id:
                            raw_user2id[uid] = len(raw_user2id)
                            self.user_id2name[uid] = uid
                    for exp in self.exp_data:
                        reviewer = exp.get("reviewerID")
                        if reviewer not in raw_user2id:
                            raw_user2id[reviewer] = len(raw_user2id)
                            self.user_id2name[reviewer] = reviewer
                    print(f"[WARN] NoAttack 模式下，动态构建了 {len(raw_user2id)} 个用户映射")
                else:
                    raise FileNotFoundError(
                        f"Missing poisoned mapping files: {idx_path} or {name_path}"
                    )
            else:
                raw_user2id       = load_pickle(idx_path)
                self.user_id2name = load_pickle(name_path)



        # —— 只在 val/test 模式下，统一保留所有任务会用到的用户映射 —— 
        if atk_snake not in ("none", "noattack") and self.mode in ("val", "test"):
            keep_users = set()
            # sequential/direct 都是从 sequential_data 拿 user_id
            if "sequential" in self.task_list or "direct" in self.task_list:
                keep_users |= { line.split()[0] for line in self.sequential_data }
            # explanation 用到 exp_data 里的 reviewerID
            if "explanation" in self.task_list:
                keep_users |= { exp.get("reviewerID") for exp in self.exp_data }

            # 过滤原始映射，只留下真正会被 __getitem__ 访问到的用户
            raw_user2id = { u: raw_user2id[u] for u in keep_users if u in raw_user2id }
            self.user_id2name = { u: self.user_id2name[u] for u in keep_users if u in self.user_id2name }

            # 重新给过滤后的用户打连续索引
            new_raw_user2id = {}
            new_user_id2name = {}
            for new_idx, u in enumerate(sorted(raw_user2id.keys())):
                new_raw_user2id[u] = new_idx
                new_user_id2name[u] = self.user_id2name[u]
            raw_user2id = new_raw_user2id
            self.user_id2name = new_user_id2name



        # 5.1) 构建 user2id 和 user_list once after filtering
        self.user2id = { str(k): v for k, v in raw_user2id.items() }
        self.user_list = [None] * len(self.user2id)
        for uid, uidx in self.user2id.items():
            self.user_list[uidx] = uid

        # 5.2) 构建 direct 任务的“有效用户”列表
        self.direct_user_list = [
            uid for uid in self.user_list
            if uid in self.user_items and len(self.user_items[uid]) > 0
        ]



        # 6) 加载 datamaps.json，只取 item2id 和 id2item
        datamaps = load_json(os.path.join(self.data_root, self.split, "datamaps.json"))
        self.item2id = datamaps["item2id"]
        self.id2item = datamaps["id2item"]

        # 7) 根据 user_id2name 的 key 类型，确定转换函数
        if self.user_id2name:
            first_key = next(iter(self.user_id2name))
            self.key_convert = int if isinstance(first_key, int) else str
        else:
            self.key_convert = str

        # 8) 加载 meta.json.gz -> meta_data, 构建 meta_dict
        self.meta_data = [m for m in parse(os.path.join(self.data_root, self.split, 'meta.json.gz'))]
        self.meta_dict = { item['asin']: idx for idx, item in enumerate(self.meta_data) }

        # 9) 加载 item2img_dict.pkl
        self.item2img_dict = load_pickle(os.path.join(self.data_root, self.split, 'item2img_dict.pkl'))

        # 准备 datum_info 用于 __getitem__
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
                # 只用 direct_user_list 的长度来计算采样数，跳过那些根本没历史的用户
                valid_n = len(self.direct_user_list)
                # 第一组模板
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    count = valid_n * self.sample_numbers[key][0]
                    self.total_length += count
                    for i in range(count):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # 第二组模板
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    count = valid_n * self.sample_numbers[key][1]
                    self.total_length += count
                    for i in range(count):
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

                # —— 更健壮的切片策略 —— 
                # 把 sequence 分解成 user_id + item 列表
                _, *items = sequence
                L = len(items)

                # 0 个 item：随机从全集里取一个当 target，history 保持空
                if L == 0:
                    history = []
                    target  = random.choice(self.all_item)
                elif L == 1:
                    # 只有 1 个 item：history 仍然留空，target=它自己
                    history = []
                    target  = items[0]
                else:
                    # >=2 个 item：随机选择 history 长度 hlen ∈ [1, min(6, L-1)]
                    max_h = min(6, L - 1)
                    hlen  = random.randint(1, max_h)
                    end_idx   = random.randint(hlen - 1, L - 2)
                    start_idx = end_idx - hlen + 1
                    history   = items[start_idx : end_idx + 1]
                    target    = items[end_idx + 1]

                purchase_history = [str(x) for x in history]
                target_item      = str(target)

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

            # 从 direct_user_list（而不是全 user_list）拿 user_id
            user_id = self.direct_user_list[datum_idx]
            # 保证 key_conversion 正确
            uid = self.key_convert(user_id)
            # 这个 user_id 一定在 user_items 里
            seq_items = self.user_items.get(user_id, [])
            # 全部转成字符串
            sequence = [str(it) for it in seq_items]
            uid = self.key_convert(user_id)
            if uid not in self.user_id2name:
                print(f"[WARN] 用户ID {uid} 不在映射中，使用默认 placeholder")
                user_desc = f"synthetic_user_{uid}"
            else:
                user_desc = self.user_id2name[uid]



            if self.mode == 'train':


                # target_candidates = sequence[1:-2]
                # target_idx = random.randint(0, len(target_candidates) - 1)
                # target_item = target_candidates[target_idx]


                # 先尝试正常切片
                target_candidates = sequence[1:-2]
                # 如果没有候选，就做最宽松的 fallback：
                #  - 至少拿倒数第二个（若长度>=2），否则直接拿最后一个
                if not target_candidates:
                    if len(sequence) >= 2:
                        target_candidates = [sequence[-2]]
                    else:
                        target_candidates = [sequence[-1]]
                # 随机挑一个（此时列表至少有 1 个元素）
                target_item = random.choice(target_candidates)


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
