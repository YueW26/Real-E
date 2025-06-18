
import os
import torch
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import json
import csv

from models.handler import train, test
from utils.data_utils import load_and_process_dataset, print_dataset_info

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Example:
# python main_tggc.py --wandb --dataset France_processed_0 --wandb_project 'NEWFRANCE' --epoch 1

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)

available_datasets = [
    'ECG_data', 'Electricity', 'Solar', 
    'France_processed_0', 'Germany_processed_0'
]
parser.add_argument('--dataset', type=str, default='ECG_data', 
                    choices=available_datasets,
                    help=f'Dataset name, options: {", ".join(available_datasets)}')

parser.add_argument('--show_datasets', action='store_true', help='Show all available dataset info')

# Hyperparameter search
parser.add_argument('--hyperparameter_search', action='store_true', help='Enable hyperparameter search')

parser.add_argument('--window_size', type=int, default=96)
parser.add_argument('--horizon', type=int, default=96)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=10)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=float, default=0.2)

# WandB
parser.add_argument('--runs', type=int, default=1, help='Number of experiment runs')
parser.add_argument('--wandb', action='store_true', help='Use wandb')
parser.add_argument('--wandb_project', type=str, default='TGGCECG', help='Wandb project name')
parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')

args = parser.parse_args()

# Show dataset info if requested
if args.show_datasets:
    print_dataset_info()
    exit(0)

# Dataset file check
file_name = args.dataset + '.csv'
data_path = os.path.join('dataset', file_name)
if not os.path.exists(data_path):
    print(f"[ERROR] Dataset file not found: {data_path}")
    print(f"Available datasets: {', '.join(available_datasets)}")
    exit(1)

# Generate experiment name if not provided
if args.experiment_name is None:
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'hypersearch' if args.hyperparameter_search else 'run'
    args.experiment_name = f"{args.dataset}_{tag}_{time_stamp}"

print(f"[INFO] Using dataset: {args.dataset}")
print(f"[INFO] Dataset path: {data_path}")
print(f"[INFO] Experiment name: {args.experiment_name}")

# === WandB Setup ===
wandb_run = None
if args.wandb:
    try:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args),
            tags=[args.dataset, 'TGGC']
        )
        print("[INFO] Wandb initialized.")
    except ImportError:
        print("[WARN] Wandb not installed. Please run: pip install wandb")
        args.wandb = False
    except Exception as e:
        print(f"[WARN] Wandb init failed: {e}")
        args.wandb = False

# === Load dataset ===
print("[INFO] Loading dataset...")
dataset_result = load_and_process_dataset(
    root_path='./dataset',
    data_file=file_name,
    target_column=None,
    features='M',
    seq_len=args.window_size,
    label_len=args.window_size // 2,
    pred_len=args.horizon,
    scale_to_01=True,
    batch_size=args.batch_size,
    freq='h',
    timeenc=0
)
train_data = dataset_result['train_dataset'].data_x
valid_data = dataset_result['val_dataset'].data_x
test_data = dataset_result['test_dataset'].data_x

# === Create result dirs ===
result_train_file = os.path.join('output', args.dataset, 'train')
result_test_file = os.path.join('output', args.dataset, 'test')
os.makedirs(result_train_file, exist_ok=True)
os.makedirs(result_test_file, exist_ok=True)

# === Run experiment ===
print("[INFO] Starting training and evaluation...")
from models.tggc_model import Model  # ensure TGGC is used
results = {}

if args.train:
    results['train'], _ = train(train_data, valid_data, args, result_train_file, wandb_run)

if args.evaluate:
    results['test'] = test(test_data, args, result_train_file, result_test_file, wandb_run)

if wandb_run:
    wandb_run.finish()

print("[INFO] Done.")
print(json.dumps(results, indent=2))