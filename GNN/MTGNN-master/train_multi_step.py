import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet

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


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    if dataset_name == 'GERMANY':
        return {
            'data': 'data/GERMANY',
            'num_nodes': 16,
            'in_dim': 2,
            'seq_in_len': 12,
            'seq_out_len': 12,
            'adj_data': 'data/sensor_graph/adj_mx_germany.pkl',  # Use Germany-specific adjacency matrix
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
            'seq_in_len': 48,
            'seq_out_len': 48,
            'adj_data': 'data/sensor_graph/adj_mx_france.pkl',  # Use France-specific adjacency matrix
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
            'seq_in_len': 12,
            'seq_out_len': 12,
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


args = parser.parse_args()

# Apply dataset-specific configuration
dataset_config = get_dataset_config(args.dataset)
for key, value in dataset_config.items():
    if hasattr(args, key) and value is not None:
        setattr(args, key, value)

torch.set_num_threads(3)


class CustomScaler:
    """Custom scaler for datasets already in 0-1 range"""
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
    
    def transform(self, data):
        return data
    
    def inverse_transform(self, data):
        # Add debug info for the first few calls
        if hasattr(self, 'debug_calls'):
            self.debug_calls += 1
        else:
            self.debug_calls = 1
            print(f"CustomScaler: Data range [{data.min():.4f}, {data.max():.4f}], shape: {data.shape}")
        return data


def improved_metric(pred, real, epsilon=1e-8):
    """Improved metric calculation with better MAPE handling for small values"""
    # Convert to numpy for easier handling
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(real, torch.Tensor):
        real = real.detach().cpu().numpy()
    
    # MAE
    mae = np.mean(np.abs(pred - real))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - real) ** 2))
    
    # Improved MAPE - add epsilon to avoid division by zero
    # Only calculate MAPE for values > epsilon
    mask = np.abs(real) > epsilon
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((pred[mask] - real[mask]) / real[mask]))
    else:
        mape = float('inf')  # All values are too small
    
    return mae, mape, rmse


def main(runid):
    # Set random seeds for reproducibility - use different seed for each run
    seed = args.seed + runid  # Different seed for each run
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    print(f"Run {runid}: Using seed {seed}")
    
    #load data
    device = torch.device(args.device)
    
    if args.custom_scaler:
        # For Germany and France datasets that are already in 0-1 range
        dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
        # Replace the scaler with custom scaler that doesn't normalize
        scaler = CustomScaler()
        dataloader['scaler'] = scaler
        
        # Don't apply StandardScaler transformation since data is already normalized
        # We need to revert the transformation applied in load_dataset
        for category in ['train', 'val', 'test']:
            # Get original data
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
    if args.adj_data is not None:
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    else:
        # Create identity matrix for datasets without predefined adjacency
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

    print(f"=== Training MTGNN on {args.dataset} Dataset ===")
    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1,args.epochs+1):
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
            if iter%args.step_size2==0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty,id)
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
            # For multi-step prediction, use all output timesteps
            metrics = engine.eval(testx, testy)
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

        if mvalid_loss<minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +"_" + args.dataset + ".pth")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + "_" + args.dataset + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)  # Remove [:,0,:,:] to keep all timesteps

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    # For validation, average over all timesteps
    pred_val = scaler.inverse_transform(yhat)
    real_val = realy[:, 0, :, :]  # Only use first feature dimension for evaluation
    pred_val_first = pred_val[:, 0, :, :] if pred_val.dim() == 4 else pred_val
    vmae, vmape, vrmse = improved_metric(pred_val_first, real_val)

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)  # Remove [:, 0, :, :] to keep all timesteps

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    
    print(f"Debug: yhat shape: {yhat.shape}, realy shape: {realy.shape}")

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        if yhat.dim() == 4:  # [batch, features, nodes, timesteps]
            pred = scaler.inverse_transform(yhat[:, :, :, i])
            real = realy[:, :, :, i]
        else:  # [batch, nodes, timesteps]
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, 0, :, i]  # Only first feature
        
        # Only use first feature for evaluation if needed
        if pred.dim() == 3:
            pred_first = pred[:, 0, :]
        else:
            pred_first = pred
            
        if real.dim() == 3:
            real_first = real[:, 0, :]
        else:
            real_first = real
        
        metrics = improved_metric(pred_first, real_first)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return vmae, vmape, vrmse, mae, mape, rmse

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    smape = np.std(mape,0)
    srmse = np.std(rmse,0)

    print(f'\n\nResults for {args.runs} runs on {args.dataset} dataset\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    
    # Adapt horizon display based on dataset
    if args.dataset == 'FRANCE':
        horizons = [2, 11, 23, 47]  # For 48 sequence length
    elif args.dataset == 'GERMANY':
        horizons = [2, 5, 11]  # For 12 sequence length
    else:
        horizons = [2, 5, 11]  # For METR-LA 12 sequence length
    
    for i in horizons:
        if i < args.seq_out_len:
            log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
            print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))





