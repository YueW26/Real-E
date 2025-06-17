import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import os
import csv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=96,help='input sequence length')         
parser.add_argument('--pred_length',type=int,default=96,help='prediction length (output sequence length)')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--run_multiple_experiments',action='store_true',help='run experiments with different sequence lengths')

args = parser.parse_args()

# Auto-detect dataset type and configure parameters accordingly
def configure_dataset_params(args):
    """
    Configure dataset-specific parameters based on data path
    """
    data_path = args.data.upper()
    
    if 'FRANCE' in data_path:
        # Original France dataset parameters
        args.num_nodes = 10
        args.adjdata = 'data/sensor_graph/adj_mx_france.pkl'
        args.save = './garage/france/'
        print(f"✅ 检测到原始France数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    elif 'GERMANY' in data_path:
        # Germany dataset parameters
        args.num_nodes = 16
        args.adjdata = 'data/sensor_graph/adj_mx_germany.pkl'
        args.save = './garage/germany/'
        print(f"✅ 检测到Germany数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    elif 'METR' in data_path:
        # METR-LA dataset parameters
        args.num_nodes = 207
        args.adjdata = 'data/sensor_graph/adj_mx.pkl'
        args.save = './garage/metr/'
        print(f"✅ 检测到METR-LA数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    elif 'BAY' in data_path:
        # PEMS-BAY dataset parameters  
        args.num_nodes = 325
        args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'
        args.save = './garage/bay/'
        print(f"✅ 检测到PEMS-BAY数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    else:
        print(f"⚠️  未识别的数据集: {data_path}")
        print(f"使用默认配置: 节点数={args.num_nodes}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save, exist_ok=True)
    
    return args

# Configure parameters based on dataset
args = configure_dataset_params(args)

def run_experiments_with_different_seq_lengths():
    """
    Run experiments with different seq_length values and save results to CSV
    """
    # seq_lengths = [6, 12, 48, 96]
    seq_lengths = [6, 12]
    results = []
    
    print("Starting experiments with different sequence lengths...")
    print(f"Sequence lengths to test: {seq_lengths}")
    print("Note: pred_length will be set equal to seq_length for each experiment")
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Starting experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"{'='*60}")
        
        # Modify args for this experiment - both seq_length and pred_length should be equal
        args.seq_length = seq_len
        args.pred_length = seq_len  # Set pred_length equal to seq_length
        args.expid = seq_len  # Use seq_length as experiment id
        
        # Generate data for this specific seq_length and pred_length
        print(f"🔄 为 seq_length={seq_len}, pred_length={seq_len} 生成数据...")
        generate_data_for_seq_length(seq_len, seq_len)  # Both parameters are equal
        
        # Run the main training function
        experiment_start_time = time.time()
        result = main_experiment()
        experiment_end_time = time.time()
        
        # Store results
        result['seq_length'] = seq_len
        result['pred_length'] = seq_len  # Also store pred_length
        result['total_experiment_time'] = experiment_end_time - experiment_start_time
        results.append(result)
        
        print(f"Completed experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"Experiment time: {experiment_end_time - experiment_start_time:.4f} seconds")
    
    # Save results to CSV
    save_results_to_csv(results)
    print(f"\nAll experiments completed! Results saved to 'experiment_results.csv'")
    
    return results

def generate_data_for_seq_length(seq_length, pred_length):
    """
    Generate dataset for specific sequence length based on the dataset type
    """
    import subprocess
    import sys
    
    # Auto-detect dataset type based on args.data
    data_path = args.data.upper()
    
    if 'FRANCE' in data_path:
        dataset_name = 'FRANCE'
        process_script = 'process_france_with_dataloader.py'
        data_file = f'data/FRANCE/train.npz'
    elif 'GERMANY' in data_path:
        dataset_name = 'GERMANY'
        process_script = 'process_germany_with_dataloader.py'
        data_file = f'data/GERMANY/train.npz'
    else:
        print(f"⚠️  不支持的数据集类型: {data_path}")
        print("跳过数据生成，使用现有数据...")
        return
    
    # Check if data already exists for this configuration
    regenerate = True
    
    if os.path.exists(data_file):
        # Check if existing data has correct seq_length
        try:
            data = np.load(data_file)
            existing_seq_len = data['x'].shape[1]  # Shape: [samples, seq_len, nodes, features]
            if existing_seq_len == seq_length:
                print(f"✅ {dataset_name}数据已存在且序列长度匹配 (seq_length={seq_length})")
                regenerate = False
            else:
                print(f"🔄 现有{dataset_name}数据序列长度不匹配 ({existing_seq_len} != {seq_length})，重新生成...")
        except Exception as e:
            print(f"⚠️  检查现有{dataset_name}数据时出错: {e}")
    
    if regenerate:
        print(f"🔄 生成{dataset_name}数据: seq_length={seq_length}, pred_length={pred_length}...")
        cmd = [
            sys.executable, process_script,
            '--step', 'process',
            '--seq_length', str(seq_length),
            '--pred_length', str(pred_length)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {dataset_name}数据生成完成")
            # Print last few lines of output for verification
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dataset_name}数据生成失败: {e}")
            print(f"错误输出: {e.stderr}")
            raise

def main_experiment():
    """
    Modified main function that returns test results instead of just printing them
    """
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, pred_length=args.pred_length)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1,3)
    #     with torch.no_grad():
    #         preds = engine.model(testx).transpose(1,3)
        
    #     if len(preds.shape) == 4:
    #         # 如果有4个维度，取第一个特征
    #         preds = preds[:, 0, :, :]
    #     outputs.append(preds)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        try:
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)

            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            
            if len(preds.shape) == 4:
                preds = preds[:, 0, :, :]
            outputs.append(preds)

            # 主动释放无用变量，清缓存
            del testx, preds
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if args.batch_size > 1:
                    args.batch_size = args.batch_size // 2
                    print(f"⚠️ 显存不足，尝试减小 batch_size 至 {args.batch_size}")
                print(f"⚠️ 第 {iter} 批次推理时显存不足，跳过该批次")
                torch.cuda.empty_cache()
            else:
                raise e
        
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    amae = []
    amape = []
    armse = []
    horizon_results = []
    
    for i in range(args.pred_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        
        # Store individual horizon results
        horizon_results.append({
            'horizon': i+1,
            'mae': metrics[0],
            'mape': metrics[1],
            'rmse': metrics[2]
        })

    avg_mae = np.mean(amae)
    avg_mape = np.mean(amape)
    avg_rmse = np.mean(armse)
    
    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(args.pred_length, avg_mae, avg_mape, avg_rmse))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

    # Return results for CSV storage
    return {
        'valid_loss_best': his_loss[bestid],
        'avg_train_time_per_epoch': np.mean(train_time),
        'avg_inference_time': np.mean(val_time),
        'test_mae_avg': avg_mae,
        'test_mape_avg': avg_mape,
        'test_rmse_avg': avg_rmse,
        'horizon_results': horizon_results
    }

def save_results_to_csv(results):
    """
    Save experiment results to CSV file
    """
    # Prepare data for CSV
    csv_data = []
    
    for result in results:
        # Basic experiment info
        row = {
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_experiment_time': result['total_experiment_time'],
            'valid_loss_best': result['valid_loss_best'],
            'avg_train_time_per_epoch': result['avg_train_time_per_epoch'],
            'avg_inference_time': result['avg_inference_time'],
            'test_mae_avg': result['test_mae_avg'],
            'test_mape_avg': result['test_mape_avg'],
            'test_rmse_avg': result['test_rmse_avg']
        }
        
        # Add individual horizon results
        for horizon_result in result['horizon_results']:
            row[f'horizon_{horizon_result["horizon"]}_mae'] = horizon_result['mae']
            row[f'horizon_{horizon_result["horizon"]}_mape'] = horizon_result['mape']
            row[f'horizon_{horizon_result["horizon"]}_rmse'] = horizon_result['rmse']
        
        csv_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('experiment_results.csv', index=False)
    
    print(f"\nResults saved to 'experiment_results.csv'")
    print(f"Columns saved: {list(df.columns)}")
    
    # Also save a summary CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_time_hours': result['total_experiment_time'] / 3600,
            'valid_loss': result['valid_loss_best'],
            'test_mae': result['test_mae_avg'],
            'test_mape': result['test_mape_avg'],
            'test_rmse': result['test_rmse_avg']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('experiment_summary.csv', index=False)
    print(f"Summary saved to 'experiment_summary.csv'")

if __name__ == "__main__":
    total_start_time = time.time()
    
    if args.run_multiple_experiments:
        # Run experiments with different sequence lengths
        print("🔬 运行多个序列长度实验...")
        results = run_experiments_with_different_seq_lengths()
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\nAll experiments completed!")
        print(f"Total time for all experiments: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for result in results:
            print(f"Seq Length {result['seq_length']:2d}, Pred Length {result['pred_length']:2d}: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")
    else:
        # Run single experiment with specified parameters
        print("🏋️ 运行单个实验...")
        print(f"📊 配置: seq_length={args.seq_length}, pred_length={args.pred_length}")
        
        # Generate data for the specified seq_length and pred_length
        print(f"🔄 为 seq_length={args.seq_length}, pred_length={args.pred_length} 生成数据...")
        generate_data_for_seq_length(args.seq_length, args.pred_length)
        
        result = main_experiment()
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\nExperiment completed!")
        print(f"Total time: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        print(f"Results: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")
