import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version 

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        assert args.whole_word_embed
        assert args.category_embed
        from model import VIP5Tuning

        model_kwargs = {}
        model_class = VIP5Tuning

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)

        if 'p5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch-')[-1])

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')

        # 添加 GPU 索引检查代码
        if args.gpu < 0:
            raise RuntimeError(f"Invalid GPU index {args.gpu}. Check CUDA_VISIBLE_DEVICES and GPU allocation.")
        
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)
        
        # Set which part of parameters as trainable
        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters

        # Calculate the percentage of trainable parameters (%)
        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch
                
            # Train
            self.model.train()
            
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                    
                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)

            if self.args.distributed:
                dist.barrier()

            if epoch > 4:
                # Validation
                valid_results = self.evaluate_epoch(epoch=epoch)

                valid_results = reduce_dict(valid_results, average=False)
                if self.verbose and step_i % 200:
                    valid_loss = valid_results['total_loss']
                    valid_loss_count = valid_results['total_loss_count']

                    avg_valid_loss = valid_loss / valid_loss_count
                    losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                    for name, loss in valid_results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(valid_results[name+'_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                if self.args.distributed:
                    dist.barrier()

                if self.verbose:
                    # Save
                    if avg_valid_loss < best_eval_loss:
                        best_eval_loss = avg_valid_loss
                        self.save("BEST_EVAL_LOSS")
                    self.save("Epoch%02d" % (epoch + 1))

                if self.args.distributed:
                    dist.barrier()
            else:
                # Skip validation
                print("Skip validation for Epoch%02d" % (epoch + 1))
                self.save("Epoch%02d" % (epoch + 1))
                
                if self.args.distributed:
                    dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=275)

            for step_i, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()

            if self.args.distributed:
                dist.barrier()

            return epoch_results


def main_worker(gpu, args):
    """主进程工作函数，用于分布式训练"""


    # 打印所有环境变量
    print("Distributed-related environment variables:")
    for key in ["CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]:
        print(f"{key}: {os.environ.get(key, 'Not Set')}")

    # 从环境变量中获取 local_rank 和 world_size
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 检查是否正确获取到 local_rank
    if args.local_rank == -1:
        raise RuntimeError("`LOCAL_RANK` is not properly set. Check your `torchrun` configuration.")


    # 获取可见的 GPU 列表
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set. Please specify available GPUs before launching.")
    gpu_list = [int(gpu) for gpu in visible_devices.split(",")]

    # 确定当前进程使用的 GPU
    if args.local_rank >= len(gpu_list):
        raise RuntimeError(
            f"Invalid local_rank {args.local_rank}. Only {len(gpu_list)} GPUs are visible: {gpu_list}"
        )
    args.gpu = gpu_list[args.local_rank]

    # 设置当前 GPU
    print(f"Process Launching at GPU {args.gpu}, local_rank: {args.local_rank}")
    torch.cuda.set_device(args.gpu)

    # 启用分布式模式
    args.distributed = args.distributed and args.world_size > 1  # 提供默认值以支持单机多 GPU 或单 GPU 模式
    if not args.distributed:
        print("Running in non-distributed mode")


    # 检查 CUDA 环境
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Distributed training requires GPU support.")
    

    # 检查 GPU 索引是否有效
    if args.local_rank < 0 or args.gpu < 0:
        raise RuntimeError(f"Invalid GPU index {args.gpu}. Check CUDA_VISIBLE_DEVICES and GPU allocation.")

    

    # 初始化分布式训练
    try:
        dist.init_process_group(backend='nccl', init_method='env://')
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed process group: {e}")
    
    # 打印分布式训练信息和数据集分割信息
    print(f"Launching distributed training: {args.distributed}, Total GPUs: {args.world_size}")
    print(f"Train dataset split: {args.train}, Valid dataset split: {args.valid}")


    # 创建训练和验证数据加载器
    print(f'Building train loader at GPU {gpu}')
    train_task_list = {'sequential': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8'],
        'direct': ['B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7'],
        'explanation': ['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11'],
    }
    train_sample_numbers = {'sequential': (5, 5), 'direct': (5, 5), 'explanation': 1}
    try:
        train_loader = get_loader(
            args,
            train_task_list,
            train_sample_numbers,
            split=args.train, 
            mode='train',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )
    except Exception as e: 
        raise RuntimeError(f"Failed to initialize train loader for split '{args.train}'. Task list: {train_task_list}, Args: {args}, Error: {e}")
    

    print(f'Building val loader at GPU {gpu}')
    val_task_list = {'sequential': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8'],
        'direct': ['B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7'],
        'explanation': ['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11'],
    }
    val_sample_numbers = {'sequential': (1, 1), 'direct': (1, 1), 'explanation': 1}
    try:
        val_loader = get_loader(
            args,
            val_task_list,
            val_sample_numbers,
            split=args.valid, 
            mode='val',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize val loader for split '{args.valid}'. Task list: {val_task_list}, Args: {args}, Error: {e}")
    

    # 初始化训练器并开始训练
    try:
        trainer = Trainer(args, train_loader, val_loader, train=True)
        trainer.train()
    finally:
        if args.distributed and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    """主函数，初始化参数和分布式环境"""
    
    # 打印所有环境变量
    print("Environment variables at main:", os.environ)

    # 设置 CUDNN 加速
    cudnn.benchmark = True

    # 解析命令行参数
    args = parse_args()
    args.distributed = True  # 确保启用分布式
    print(f"Parsed arguments: {args}")

    # 检查可用 GPU 数量
    ngpus_per_node = torch.cuda.device_count()
    print(f"Number of GPUs available: {ngpus_per_node}")
    args.world_size = ngpus_per_node

    # 构建损失名称列表
    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')
    args.LOSSES_NAME = LOSSES_NAME

    # 生成运行名称和评论
    comments = []
    dsets = []
    if 'toys' in args.train:
        dsets.append('toys')
    if 'beauty' in args.train:
        dsets.append('beauty')
    if 'sports' in args.train:
        dsets.append('sports')
    if 'clothing' in args.train:
        dsets.append('clothing')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    # 生成运行的时间戳
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    # 确定项目目录和运行名称
    project_dir = Path(__file__).resolve().parent.parent
    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    # 调用主进程函数
    if args.distributed:
        main_worker(args.local_rank, args)
