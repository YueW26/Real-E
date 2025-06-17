import torch
import numpy as np
import argparse
import time
import os
import subprocess
import sys
from util import *
from trainer import Trainer
from net import gtnet
import pandas as pd

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()

# Dataset selection and path configuration
parser.add_argument('--dataset', type=str, default='METR-LA', 
                   choices=['METR-LA', 'GERMANY', 'FRANCE'],
                   help='dataset to use: METR-LA, GERMANY, or FRANCE')
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')

parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')

parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--runs',type=int,default=1,help='number of runs')

# Add custom scaler parameter for datasets with 0-1 range
parser.add_argument('--custom_scaler', type=str_to_bool, default=False, 
                   help='whether to use custom scaler for 0-1 range data')

parser.add_argument('--run_multiple_experiments',action='store_true',help='run experiments with different sequence lengths')
def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    if dataset_name == 'GERMANY':
        return {
            'data': 'data/GERMANY',
            'num_nodes': 16,
            'in_dim': 2,
            'adj_data': 'data/sensor_graph/adj_mx_germany.pkl',
            'custom_scaler': True,
            'batch_size': 32,
            'subgraph_size': 8,
            'node_dim': 20,
            'conv_channels': 16,
            'residual_channels': 16,
            'skip_channels': 32,
            'end_channels': 64
        }
    elif dataset_name == 'FRANCE':
        return {
            'data': 'data/FRANCE',
            'num_nodes': 10,
            'in_dim': 2,
            'adj_data': 'data/sensor_graph/adj_mx_france.pkl',
            'custom_scaler': True,
            'batch_size': 32,
            'subgraph_size': 5,
            'node_dim': 20,
            'conv_channels': 16,
            'residual_channels': 16,
            'skip_channels': 32,
            'end_channels': 64
        }
    else:  # METR-LA (default)
        return {
            'data': 'data/METR-LA',
            'num_nodes': 207,
            'in_dim': 2,
            'adj_data': 'data/sensor_graph/adj_mx.pkl',
            'custom_scaler': False,
            'batch_size': 64,
            'subgraph_size': 20,
            'node_dim': 40,
            'conv_channels': 32,
            'residual_channels': 32,
            'skip_channels': 64,
            'end_channels': 128
        }

class CustomScaler:
    """Custom scaler for datasets already in 0-1 range"""
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
    
    def transform(self, data):
        return data
    
    def inverse_transform(self, data):
        return data

def improved_metric(pred, real, epsilon=1e-8):
    """Improved metric calculation with better MAPE handling for small values"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(real, torch.Tensor):
        real = real.detach().cpu().numpy()
    
    # MAE
    mae = np.mean(np.abs(pred - real))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - real) ** 2))
    
    # Improved MAPE - add epsilon to avoid division by zero
    mask = np.abs(real) > epsilon
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((pred[mask] - real[mask]) / real[mask]))
    else:
        mape = float('inf')
    
    return mae, mape, rmse

def generate_data_for_horizon(dataset_name, seq_in_len, seq_out_len):
    """
    Generate dataset for specific seq_in_len and seq_out_len based on dataset type
    """
    if dataset_name == 'FRANCE':
        process_script = 'process_france_with_dataloader.py'
        data_file = f'data/FRANCE/train.npz'
    elif dataset_name == 'GERMANY':
        process_script = 'process_germany_with_dataloader.py'
        data_file = f'data/GERMANY/train.npz'
    else:
        print(f"⚠️  Dataset {dataset_name} doesn't need data generation")
        return
    
    # Check if data already exists for this configuration
    regenerate = True
    
    if os.path.exists(data_file):
        try:
            data = np.load(data_file)
            existing_seq_in = data['x'].shape[1]
            existing_seq_out = data['y'].shape[1] 
            if existing_seq_in == seq_in_len and existing_seq_out == seq_out_len:
                print(f"✅ {dataset_name} data exists with correct configuration (seq_in={seq_in_len}, seq_out={seq_out_len})")
                regenerate = False
            else:
                print(f"🔄 {dataset_name} data configuration mismatch ({existing_seq_in},{existing_seq_out} != {seq_in_len},{seq_out_len}), regenerating...")
        except Exception as e:
            print(f"⚠️  Error checking existing {dataset_name} data: {e}")
    
    if regenerate:
        print(f"🔄 Generating {dataset_name} data: seq_in_len={seq_in_len}, seq_out_len={seq_out_len}...")
        cmd = [
            sys.executable, process_script,
            '--step', 'process',
            '--seq_length', str(seq_in_len),
            '--pred_length', str(seq_out_len)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {dataset_name} data generation completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dataset_name} data generation failed: {e}")
            print(f"Error output: {e.stderr}")
            raise

def main(runid):
    """
    Main training function
    """
    print(f"\n{'='*80}")
    print(f"Training MTGNN on {args.dataset} - seq_in_len: {args.seq_in_len}, seq_out_len: {args.seq_out_len}")
    print(f"Run ID: {runid}")
    print(f"{'='*80}")
    
    # Set random seeds for reproducibility
    seed = args.seed + runid
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    print(f"Run {runid}: Using seed {seed}")
    
    # Generate data for this specific configuration
    generate_data_for_horizon(args.dataset, args.seq_in_len, args.seq_out_len)
    
    # Load data
    device = torch.device(args.device)
    
    if args.custom_scaler:
        # For Germany and France datasets that are already in 0-1 range
        dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
        scaler = CustomScaler()
        dataloader['scaler'] = scaler
        
        # Don't apply StandardScaler transformation since data is already normalized
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(args.data, category + '.npz'))
            dataloader['x_' + category] = cat_data['x']
            dataloader['y_' + category] = cat_data['y']
        
        # Recreate data loaders with original data
        dataloader['train_loader'] = DataLoaderM(dataloader['x_train'], dataloader['y_train'], args.batch_size)
        dataloader['val_loader'] = DataLoaderM(dataloader['x_val'], dataloader['y_val'], args.batch_size)
        dataloader['test_loader'] = DataLoaderM(dataloader['x_test'], dataloader['y_test'], args.batch_size)
    else:
        # For METR-LA dataset
        dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
        scaler = dataloader['scaler']

    # Handle adjacency matrix
    if args.adj_data is not None and os.path.exists(args.adj_data):
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    else:
        predefined_A = torch.zeros(args.num_nodes, args.num_nodes)
    
    predefined_A = predefined_A.to(device)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    # Adjust step_size1 based on sequence length to prevent task_level from exceeding seq_out_len
    # For curriculum learning, ensure task_level grows appropriately
    # Use a larger step_size1 for shorter sequences to prevent task_level from growing too fast
    if args.seq_out_len <= 12:
        adjusted_step_size1 = args.step_size1 * 2  # Double the step size for short sequences
    else:
        adjusted_step_size1 = args.step_size1
    print(f"Original step_size1: {args.step_size1}, Adjusted step_size1: {adjusted_step_size1} (seq_out_len: {args.seq_out_len})")

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, adjusted_step_size1, args.seq_out_len, scaler, device, args.cl)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            
            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            
            num_sub = int(args.num_nodes / args.num_split)
            for j in range(args.num_split):
                if j != args.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                
                metrics = engine.train(tx, ty, id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        
        t2 = time.time()
        train_time.append(t2 - t1)
        
        # Validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        
        s2 = time.time()
        val_time.append(s2 - s1)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if mvalid_loss < minl:
            model_path = args.save + f"exp{args.expid}_{runid}_{args.dataset}_seq{args.seq_in_len}_pred{args.seq_out_len}.pth"
            torch.save(engine.model.state_dict(), model_path)
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    model_path = args.save + f"exp{args.expid}_{runid}_{args.dataset}_seq{args.seq_in_len}_pred{args.seq_out_len}.pth"
    engine.model.load_state_dict(torch.load(model_path))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # Test data evaluation
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    
    # Evaluate all timesteps and take average
    if yhat.dim() == 4:  # [batch, features, nodes, timesteps]
        pred = scaler.inverse_transform(yhat)
        real = realy
    else:  # [batch, nodes, timesteps]
        pred = scaler.inverse_transform(yhat)
        real = realy[:, 0, :, :]  # Only first feature
    
    # Only use first feature for evaluation if needed
    if pred.dim() == 4:
        pred_first = pred[:, 0, :, :]  # [batch, nodes, timesteps]
    else:
        pred_first = pred
        
    if real.dim() == 4:
        real_first = real[:, 0, :, :]  # [batch, nodes, timesteps]
    else:
        real_first = real
    
    # Average over all timesteps
    mae, mape, rmse = improved_metric(pred_first, real_first)
    
    return mae, mape, rmse

def run_multiple_seq_lengths():
    """
    Run experiments with different sequence lengths and output results
    """
    # Test configurations: seq_in_len = seq_out_len
    test_configs = [6, 12]
    
    print(f"\n{'='*80}")
    print(f"Running Multiple Sequence Length Experiments on {args.dataset}")
    print(f"Testing sequence lengths: {test_configs}")
    print(f"Number of runs per configuration: {args.runs}")
    print(f"{'='*80}")
    
    all_results = []
    
    for seq_len in test_configs:
        print(f"\n{'='*60}")
        print(f"Starting experiments for Seq Length {seq_len}, Pred Length {seq_len}")
        print(f"{'='*60}")
        
        # Update args with current configuration
        original_seq_in = args.seq_in_len
        original_seq_out = args.seq_out_len
        
        args.seq_in_len = seq_len
        args.seq_out_len = seq_len
        
        mae_list = []
        mape_list = []
        rmse_list = []
        
        for run in range(args.runs):
            print(f"\n--- Run {run + 1}/{args.runs} for Seq Length {seq_len} ---")
            
            mae, mape, rmse = main(run)
            
            mae_list.append(mae)
            mape_list.append(mape)
            rmse_list.append(rmse)
        
        # Calculate averages
        final_mae = np.mean(mae_list)
        final_mape = np.mean(mape_list)
        final_rmse = np.mean(rmse_list)
        
        all_results.append({
            'seq_len': seq_len,
            'mae': final_mae,
            'mape': final_mape,
            'rmse': final_rmse,
            'mae_std': np.std(mae_list),
            'mape_std': np.std(mape_list),
            'rmse_std': np.std(rmse_list)
        })
        
        # Print results for this configuration
        print(f"\n--- Results for Seq Length {seq_len} (averaged over {args.runs} runs) ---")
        print(f'Seq Length {seq_len:2d}, Pred Length {seq_len:2d}: MAE={final_mae:.4f}, MAPE={final_mape:.4f}, RMSE={final_rmse:.4f}')
        if args.runs > 1:
            print(f'Standard deviation: MAE={np.std(mae_list):.4f}, MAPE={np.std(mape_list):.4f}, RMSE={np.std(rmse_list):.4f}')
        
        # Restore original values
        args.seq_in_len = original_seq_in
        args.seq_out_len = original_seq_out
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS FOR ALL SEQUENCE LENGTHS ON {args.dataset}")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)

    # 保存为 CSV 文件
    df.to_csv(f'{args.dataset}_results.csv', index=False)
    print(f"Results saved to {args.dataset}_results.csv")

    for result in all_results:
        seq_len = result['seq_len']
        mae = result['mae']
        mape = result['mape']
        rmse = result['rmse']
        mae_std = result['mae_std']
        mape_std = result['mape_std']
        rmse_std = result['rmse_std']
        
        print(f'Seq Length {seq_len:2d}, Pred Length {seq_len:2d}: MAE={mae:.4f}, MAPE={mape:.4f}, RMSE={rmse:.4f}')
        if args.runs > 1:
            print(f'Standard deviation: MAE={mae_std:.4f}, MAPE={mape_std:.4f}, RMSE={rmse_std:.4f}')
    
    return all_results

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Apply dataset-specific configuration
    dataset_config = get_dataset_config(args.dataset)
    for key, value in dataset_config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    torch.set_num_threads(3)
    
    if args.run_multiple_experiments:
        run_multiple_seq_lengths()
    else:
        mae_list = []
        mape_list = []
        rmse_list = []
        
        for i in range(args.runs):
            mae, mape, rmse = main(i)
            mae_list.append(mae)
            mape_list.append(mape)
            rmse_list.append(rmse)

        # Output simple results
        final_mae = np.mean(mae_list)
        final_mape = np.mean(mape_list)
        final_rmse = np.mean(rmse_list)
        
        print(f'\n{"="*80}')
        print(f'Final Results ({args.runs} runs)')
        print(f'{"="*80}')
        print(f'seq_in_len {args.seq_in_len}, seq_out_len {args.seq_out_len}: MAE={final_mae:.4f}, MAPE={final_mape:.4f}, RMSE={final_rmse:.4f}')
        
        if args.runs > 1:
            print(f'Standard deviation: MAE={np.std(mae_list):.4f}, MAPE={np.std(mape_list):.4f}, RMSE={np.std(rmse_list):.4f}') 