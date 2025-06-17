import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
import numpy as np
import json
import csv
from utils.data_utils import load_and_process_dataset, print_dataset_info


#  python main.py --wandb --dataset France_processed_0 --wandb_project 'NEWFRANCE' --epoch 1
#  python main.py --wandb --dataset Germany_processed_0 --wandb_project 'StemGNN_Germany'
#  python main.py --hyperparameter_search --dataset ECG_data --epoch 10

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)

# 更新数据集参数，添加所有可用数据集的支持
available_datasets = [
    'ECG_data', 'ECG_data_0', 'PeMS07', 
    'France_processed_0', 'Germany_processed_0'
]
parser.add_argument('--dataset', type=str, default='ECG_data', 
                   choices=available_datasets,
                   help=f'数据集选择，可选: {", ".join(available_datasets)}')

parser.add_argument('--show_datasets', action='store_true', help='显示所有可用数据集信息')

# 超参数搜索选项
parser.add_argument('--hyperparameter_search', action='store_true', help='进行window_size和horizon的超参数搜索')

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
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)


# 新增 wandb 相关参数
parser.add_argument('--runs', type=int, default=1, help='Number of runs to perform')
parser.add_argument('--wandb', action='store_true', help='Use wandb for experiment tracking')
parser.add_argument('--wandb_project', type=str, default='StemGNNECG', help='Wandb project name')
parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for organizing results')

args = parser.parse_args()

# 如果用户要求显示数据集信息，则显示并退出
if args.show_datasets:
    print_dataset_info()
    exit(0)

# 检查数据集文件是否存在
data_file = args.dataset + '.csv'
data_path = os.path.join('dataset', data_file)
if not os.path.exists(data_path):
    print(f"❌ 数据集文件不存在: {data_path}")
    print(f"可用的数据集: {', '.join(available_datasets)}")
    print("\n使用 --show_datasets 查看详细信息")
    exit(1)

print(f"✓ 使用数据集: {args.dataset}")
print(f"✓ 数据集文件: {data_path}")

# 设置实验名称
if args.experiment_name is None:
    if args.hyperparameter_search:
        args.experiment_name = f"{args.dataset}_hypersearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        args.experiment_name = f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f'Training configs: {args}')

# 超参数搜索函数
def hyperparameter_search():
    """执行超参数搜索"""
    print("\n" + "="*60)
    print("开始超参数搜索: window_size 和 horizon")
    print("="*60)
    
    # 定义搜索空间
    search_space = [6, 12, 48]
    results = []
    
    # 为每个window_size生成有效的horizon组合
    valid_combinations = []
    for window_size in search_space:
        for horizon in search_space:
            if window_size >= horizon:  # window_size不能比horizon小
                valid_combinations.append((window_size, horizon))
    
    print(f"有效的参数组合: {valid_combinations}")
    print(f"总共要测试 {len(valid_combinations)} 种组合\n")
    
    # 初始化 wandb（如果启用）
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project + "_hypersearch",
                name=args.experiment_name,
                config=vars(args),
                tags=[args.dataset, "hyperparameter_search"]
            )
            print("✓ Wandb 超参数搜索初始化成功")
        except ImportError:
            print("⚠ Wandb 未安装，请运行: pip install wandb")
            args.wandb = False
        except Exception as e:
            print(f"⚠ Wandb 初始化失败: {e}")
            args.wandb = False
    
    # 遍历每个组合
    for i, (window_size, horizon) in enumerate(valid_combinations):
        print(f"\n[{i+1}/{len(valid_combinations)}] 测试 window_size={window_size}, horizon={horizon}")
        print("-" * 40)
        
        # 更新args中的参数
        args.window_size = window_size
        args.horizon = horizon
        
        try:
            # 重新加载数据集（因为window_size和horizon变了）
            dataset_result = load_and_process_dataset(
                root_path='./dataset',
                data_file=data_file,
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
            
            # 提取数据
            train_dataset = dataset_result['train_dataset']
            val_dataset = dataset_result['val_dataset'] 
            test_dataset = dataset_result['test_dataset']
            
            train_data = train_dataset.data_x
            valid_data = val_dataset.data_x
            test_data = test_dataset.data_x
            
            print(f"数据加载完成: 训练集{train_data.shape}, 验证集{valid_data.shape}, 测试集{test_data.shape}")
            
            # 设置结果目录
            result_train_file = os.path.join('output', 'hypersearch', args.dataset, f'ws{window_size}_hz{horizon}', 'train')
            result_test_file = os.path.join('output', 'hypersearch', args.dataset, f'ws{window_size}_hz{horizon}', 'test')
            
            if not os.path.exists(result_train_file):
                os.makedirs(result_train_file)
            if not os.path.exists(result_test_file):
                os.makedirs(result_test_file)
            
            # 训练
            if args.train:
                print("开始训练...")
                train_start = datetime.now()
                train_metrics, normalize_statistic = train(train_data, valid_data, args, result_train_file, wandb_run)
                train_end = datetime.now()
                train_time = (train_end - train_start).total_seconds() / 60
                print(f"训练完成，耗时: {train_time:.2f} 分钟")
            
            # 测试
            if args.evaluate:
                print("开始测试...")
                test_start = datetime.now()
                test_metrics = test(test_data, args, result_train_file, result_test_file, wandb_run)
                test_end = datetime.now()
                test_time = (test_end - test_start).total_seconds() / 60
                print(f"测试完成，耗时: {test_time:.2f} 分钟")
                
                # 保存结果
                result = {
                    'window_size': window_size,
                    'horizon': horizon,
                    'mae': test_metrics.get('mae', 0),
                    'rmse': test_metrics.get('rmse', 0),
                    'mape': test_metrics.get('mape', 0),
                    'train_time_min': round(train_time, 2),
                    'test_time_min': round(test_time, 2)
                }
                results.append(result)
                
                # 打印当前结果
                print(f"Seq Length {window_size:2d}, Pred Length {horizon:2d}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
                
                # 记录到wandb
                if wandb_run:
                    wandb_run.log({
                        "window_size": window_size,
                        "horizon": horizon,
                        "test_mae": result['mae'],
                        "test_rmse": result['rmse'],
                        "test_mape": result['mape'],
                        "train_time_min": result['train_time_min'],
                        "test_time_min": result['test_time_min'],
                        "combination_id": i
                    })
        
        except Exception as e:
            print(f"❌ 参数组合 (window_size={window_size}, horizon={horizon}) 失败: {e}")
            continue
    
    # 保存和显示所有结果
    print("\n" + "="*60)
    print("超参数搜索结果汇总")
    print("="*60)
    
    if results:
        # 按window_size分组并显示结果
        results_by_window = {}
        for result in results:
            ws = result['window_size']
            if ws not in results_by_window:
                results_by_window[ws] = []
            results_by_window[ws].append(result)
        
        # 显示格式化结果
        all_results_text = []
        for window_size in sorted(results_by_window.keys()):
            window_results = results_by_window[window_size]
            # 按MAE排序，找出最好的horizon
            window_results.sort(key=lambda x: x['mae'])
            
            print(f"\nWindow Size {window_size} 的结果:")
            for result in window_results:
                result_text = f"Seq Length {result['window_size']:2d}, Pred Length {result['horizon']:2d}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}"
                print(f"  {result_text}")
                all_results_text.append(result_text)
        
        # 保存结果到文件
        results_file = os.path.join('output', f'hyperparameter_search_results_{args.dataset}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存格式化结果到文本文件
        formatted_results_file = os.path.join('output', f'hyperparameter_search_summary_{args.dataset}.txt')
        with open(formatted_results_file, 'w') as f:
            f.write("超参数搜索结果汇总\n")
            f.write("="*40 + "\n\n")
            for result_text in all_results_text:
                f.write(result_text + "\n")
        
        print(f"\n✓ 详细结果已保存到: {results_file}")
        print(f"✓ 格式化结果已保存到: {formatted_results_file}")
        
        # 找出最好的结果
        best_result = min(results, key=lambda x: x['mae'])
        print(f"\n🏆 最佳结果:")
        print(f"Seq Length {best_result['window_size']:2d}, Pred Length {best_result['horizon']:2d}: MAE={best_result['mae']:.4f}, RMSE={best_result['rmse']:.4f}")
        
        if wandb_run:
            wandb_run.log({
                "best_window_size": best_result['window_size'],
                "best_horizon": best_result['horizon'], 
                "best_mae": best_result['mae'],
                "best_rmse": best_result['rmse'],
                "total_combinations_tested": len(results)
            })
    else:
        print("❌ 没有成功的结果")
    
    # 关闭wandb
    if wandb_run:
        wandb.finish()
    
    return results

# 主程序逻辑修改
if args.hyperparameter_search:
    # 执行超参数搜索
    hyperparameter_search()
    exit(0)

# 原有的单次实验逻辑保持不变
# 初始化 wandb（如果启用）
wandb_run = None
if args.wandb:
    try:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args),
            tags=[args.dataset]
        )
        print("✓ Wandb 初始化成功")
    except ImportError:
        print("⚠ Wandb 未安装，请运行: pip install wandb")
        args.wandb = False
    except Exception as e:
        print(f"⚠ Wandb 初始化失败: {e}")
        args.wandb = False

# 创建实验目录
if args.runs > 1:
    experiment_base_dir = os.path.join('output', args.experiment_name)
    result_train_file = os.path.join(experiment_base_dir, 'train')
    result_test_file = os.path.join(experiment_base_dir, 'test')
else:
    result_train_file = os.path.join('output', args.dataset, 'train')
    result_test_file = os.path.join('output', args.dataset, 'test')

if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

# 保存实验配置
if args.runs > 1:
    config_file = os.path.join(os.path.dirname(result_train_file), 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

# 使用 Dataset_Opennem 类加载数据集
print("\n" + "="*50)
print("开始使用Dataset_Opennem类加载数据集...")
print("="*50)
try:
    # 使用 load_and_process_dataset 函数，内部使用 CompatibleDataset_Opennem 类
    dataset_result = load_and_process_dataset(
        root_path='./dataset',
        data_file=data_file,
        target_column=None,  # 自动检测目标列
        features='M',        # 多变量预测
        seq_len=args.window_size,
        label_len=args.window_size // 2,
        pred_len=args.horizon,
        scale_to_01=True,    # 确保数据在0-1范围
        batch_size=args.batch_size,
        freq='h',
        timeenc=0
    )
    
    print("✓ Dataset_Opennem 数据集加载成功!")
    print(f"  数据集名称: {dataset_result['dataset_name']}")
    print(f"  目标列: {dataset_result['target_column']}")
    print(f"  特征维度: {dataset_result['feature_dim']}")
    print(f"  总样本数: {dataset_result['total_samples']}")
    print(f"  数据形状: {dataset_result['data_shape']}")
    
    # 从Dataset_Opennem实例中提取numpy数组数据
    # 注意：我们需要将数据转换为main.py的train/test函数期望的格式
    train_dataset = dataset_result['train_dataset']
    val_dataset = dataset_result['val_dataset'] 
    test_dataset = dataset_result['test_dataset']
    
    # 提取原始数据（已经是0-1范围）
    print("\n提取训练、验证和测试数据...")
    train_data = train_dataset.data_x  # 已经是numpy数组且在0-1范围
    valid_data = val_dataset.data_x
    test_data = test_dataset.data_x
    
    print(f"✓ 数据提取完成:")
    print(f"  训练集形状: {train_data.shape}")
    print(f"  验证集形状: {valid_data.shape}")
    print(f"  测试集形状: {test_data.shape}")
    print(f"  数据范围: [{train_data.min():.6f}, {train_data.max():.6f}]")
    print(f"  注意: Dataset_Opennem已自动按7:2:1比例划分训练/验证/测试集")
    
except Exception as e:
    print(f"❌ Dataset_Opennem 数据集加载失败: {e}")
    exit(1)

# Dataset_Opennem 已经完成了数据划分，无需手动划分
# split data
# train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
# valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
# test_ratio = 1 - train_ratio - valid_ratio
# train_data = train_data[:int(train_ratio * len(train_data))]
# valid_data = valid_data[int(train_ratio * len(valid_data)):int((train_ratio + valid_ratio) * len(valid_data))]
# test_data = test_data[int((train_ratio + valid_ratio) * len(test_data)):]

def save_results_to_csv(results_list, dataset_name, num_runs):
    """将运行结果保存到CSV文件"""
    
    # 创建文件名
    if num_runs == 1:
        filename = f"results_{dataset_name}_single_run.csv"
    else:
        filename = f"results_{dataset_name}_{num_runs}runs.csv"
    
    filepath = os.path.join('output', filename)
    
    # 确保output目录存在
    os.makedirs('output', exist_ok=True)
    
    # 定义CSV列名
    fieldnames = [
        'run_id',
        'dataset',
        'timestamp',
        'final_test_mape',
        'final_test_mae', 
        'final_test_rmse',
        'total_train_time_min',
        'total_eval_time_min',
        'best_epoch',
        'total_params',
        'train_epochs',
        'batch_size',
        'learning_rate',
        'device',
        'window_size',
        'horizon'
    ]
    
    # 检查文件是否存在
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 写入每次运行的结果，只保留fieldnames中的字段
        for result in results_list:
            # 过滤掉不在fieldnames中的字段
            filtered_result = {key: value for key, value in result.items() if key in fieldnames}
            writer.writerow(filtered_result)
    
    print(f"✓ 运行结果已保存到: {filepath}")
    return filepath

def run_single_experiment(run_id=0):
    """执行单次实验"""
    # 设置随机种子
    torch.manual_seed(run_id)
    np.random.seed(run_id)
    
    print(f"开始第 {run_id + 1}/{args.runs} 次运行")
    
    results = {
        'run_id': run_id,
        'dataset': args.dataset,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_epochs': args.epoch,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'device': args.device,
        'window_size': args.window_size,
        'horizon': args.horizon
    }
    
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            train_metrics, normalize_statistic = train(train_data, valid_data, args, result_train_file, wandb_run)
            after_train = datetime.now().timestamp()
            train_time = (after_train - before_train) / 60
            print(f'第 {run_id + 1} 次运行训练耗时: {train_time:.2f} 分钟')
            
            results['total_train_time_min'] = round(train_time, 4)
            results['train_metrics'] = train_metrics
            
            # 从训练指标中提取信息
            if 'best_epoch' in train_metrics:
                results['best_epoch'] = train_metrics['best_epoch']
            if 'total_params' in train_metrics:
                results['total_params'] = train_metrics['total_params']
            
        except KeyboardInterrupt:
            print('-' * 99)
            print('提前退出训练')
            if wandb_run:
                wandb_run.log({"interrupted": True})
            results['interrupted'] = True
    
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test_metrics = test(test_data, args, result_train_file, result_test_file, wandb_run)
        after_evaluation = datetime.now().timestamp()
        eval_time = (after_evaluation - before_evaluation) / 60
        print(f'第 {run_id + 1} 次运行评估耗时: {eval_time:.2f} 分钟')
        
        results['total_eval_time_min'] = round(eval_time, 4)
        results['test_metrics'] = test_metrics
        
        # 提取测试指标
        if isinstance(test_metrics, dict):
            results['final_test_mape'] = round(test_metrics.get('mape', 0), 6)
            results['final_test_mae'] = round(test_metrics.get('mae', 0), 6)
            results['final_test_rmse'] = round(test_metrics.get('rmse', 0), 6)
        
        # 记录最终结果到 wandb，增加更多详细信息
        if wandb_run:
            log_data = {
                "final_test_mape": results.get('final_test_mape', 0),
                "final_test_mae": results.get('final_test_mae', 0),
                "final_test_rmse": results.get('final_test_rmse', 0),
                "total_train_time_min": results.get('total_train_time_min', 0),
                "total_eval_time_min": results.get('total_eval_time_min', 0),
                "run_id": run_id,
                "dataset_name": args.dataset,
                "experiment_name": args.experiment_name
            }
            
            # 如果有节点级别的指标，也记录
            if 'mae_node' in test_metrics:
                log_data.update({
                    "final_test_mae_node_count": len(test_metrics['mae_node']),
                    "final_test_mae_node_worst": np.max(test_metrics['mae_node']),
                    "final_test_mae_node_best": np.min(test_metrics['mae_node']),
                    "final_test_mape_node_worst": np.max(test_metrics['mape_node']),
                    "final_test_mape_node_best": np.min(test_metrics['mape_node']),
                    "final_test_rmse_node_worst": np.max(test_metrics['rmse_node']),
                    "final_test_rmse_node_best": np.min(test_metrics['rmse_node']),
                })
            
            wandb_run.log(log_data)
    
    return results

if __name__ == '__main__':
    if args.runs == 1:
        # 单次运行
        torch.manual_seed(0)
        results = run_single_experiment(0)
        
        # 保存单次运行结果到CSV
        save_results_to_csv([results], args.dataset, 1)
        
        print('done')
    else:
        # 多次运行
        print(f"开始 {args.runs} 次运行的实验")
        all_results = []
        csv_results = []  # 用于保存到CSV的结果列表
        
        for run_id in range(args.runs):
            result = run_single_experiment(run_id)
            all_results.append(result)
            csv_results.append(result)  # 添加到CSV结果列表
        
        # 保存所有运行结果到CSV
        save_results_to_csv(csv_results, args.dataset, args.runs)
        
        # 计算多次运行的统计信息
        if len(all_results) > 1:
            test_mapes = [r.get('final_test_mape', 0) for r in all_results if 'final_test_mape' in r and not r.get('interrupted', False)]
            test_maes = [r.get('final_test_mae', 0) for r in all_results if 'final_test_mae' in r and not r.get('interrupted', False)]
            test_rmses = [r.get('final_test_rmse', 0) for r in all_results if 'final_test_rmse' in r and not r.get('interrupted', False)]
            
            if test_mapes:
                print(f"\n多次运行结果统计 ({len(test_mapes)} 次成功运行):")
                print("-" * 50)
                print(f"MAPE: {np.mean(test_mapes):.4f} ± {np.std(test_mapes):.4f}")
                print(f"MAE:  {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
                print(f"RMSE: {np.mean(test_rmses):.4f} ± {np.std(test_rmses):.4f}")
                
                # 记录汇总统计到 wandb
                if wandb_run:
                    wandb_run.log({
                        "summary_mape_mean": np.mean(test_mapes),
                        "summary_mape_std": np.std(test_mapes),
                        "summary_mae_mean": np.mean(test_maes),
                        "summary_mae_std": np.std(test_maes),
                        "summary_rmse_mean": np.mean(test_rmses),
                        "summary_rmse_std": np.std(test_rmses),
                        "successful_runs": len(test_mapes),
                        "total_runs": args.runs
                    })
                
                # 保存汇总结果
                summary_results = {
                    'mape': {'mean': np.mean(test_mapes), 'std': np.std(test_mapes), 'values': test_mapes},
                    'mae': {'mean': np.mean(test_maes), 'std': np.std(test_maes), 'values': test_maes},
                    'rmse': {'mean': np.mean(test_rmses), 'std': np.std(test_rmses), 'values': test_rmses},
                    'successful_runs': len(test_mapes),
                    'total_runs': args.runs
                }
                
                summary_file = os.path.join(os.path.dirname(result_train_file), 'experiment_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary_results, f, indent=2)
        
        print('实验完成!')
    
    # 关闭 wandb
    if wandb_run:
        wandb.finish()


