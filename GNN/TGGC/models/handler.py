import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from tggc_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os

import pdb

from utils.math_utils import evaluate


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_tggc.pt')  
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_tggc.pt')  
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f, weights_only=False)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float64)
            while step < horizon:
                forecast_result, a = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None, wandb_run=None, epoch=None, is_test=False):
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
    # TODO double check the normalize value 
    score = evaluate(target, forecast)
    score_by_node = evaluate(target, forecast, by_node=True)
    end = datetime.now()

    score_norm = evaluate(target_norm, forecast_norm)
    
    # 确定日志前缀
    prefix = "test" if is_test else "val"
    
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    
    # 记录到 wandb
    if wandb_run:
        log_dict = {
            f"{prefix}_mape_norm": score_norm[0],
            f"{prefix}_mae_norm": score_norm[1],
            f"{prefix}_rmse_norm": score_norm[2],
            f"{prefix}_mape": score[0],
            f"{prefix}_mae": score[1],
            f"{prefix}_rmse": score[2],
        }
        
        if epoch is not None:
            log_dict["epoch"] = epoch
            
        wandb_run.log(log_dict)
    
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2])


def train(train_data, valid_data, args, result_file, wandb_run=None):
    node_cnt = train_data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
    model.to(args.device)



    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    
    # wandb
    if wandb_run:
        wandb_run.log({"total_params": total_params})

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    
    # 
    train_losses = []
    val_maes = []
    val_mapes = []
    val_rmses = []
    
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.zero_grad()

            # pdb.set_trace()

            forecast, _ = model(inputs)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        
        avg_train_loss = loss_total / cnt
        train_losses.append(avg_train_loss)
        
        epoch_time = time.time() - epoch_start_time
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, epoch_time, avg_train_loss))
        
        save_model(model, result_file, epoch)
        
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
            
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         result_file=result_file, wandb_run=wandb_run, epoch=epoch)
                         
            val_maes.append(performance_metrics['mae'])
            val_mapes.append(performance_metrics['mape'])
            val_rmses.append(performance_metrics['rmse'])
            
            # 
            if wandb_run:
                current_lr = my_optim.param_groups[0]['lr']
                wandb_run.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time
                })
                
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
                
                # 
                if wandb_run:
                    wandb_run.log({
                        "best_val_mae": best_validate_mae,
                        "best_epoch": epoch
                    })
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                save_model(model, result_file)
        else:
            # 
            if wandb_run:
                current_lr = my_optim.param_groups[0]['lr']
                wandb_run.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time
                })
                
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            print(f"Early stopping at epoch {epoch}")
            if wandb_run:
                wandb_run.log({"early_stop_epoch": epoch})
            break
    
    # 
    training_history = {
        'train_losses': train_losses,
        'val_maes': val_maes,
        'val_mapes': val_mapes,
        'val_rmses': val_rmses,
        'best_val_mae': best_validate_mae
    }
    
    history_file = os.path.join(result_file, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # performance_metrics
    performance_metrics['best_epoch'] = args.epoch - 1  # 
    performance_metrics['total_params'] = total_params
    performance_metrics['best_val_mae'] = best_validate_mae
    
    return performance_metrics, normalize_statistic


def test(test_data, args, result_train_file, result_test_file, wandb_run=None):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    
    # 
    test_start_time = datetime.now()
    
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                      node_cnt, args.window_size, args.horizon,
                      result_file=result_test_file, wandb_run=wandb_run, is_test=True)
    
    # 
    test_end_time = datetime.now()
    test_duration = (test_end_time - test_start_time).total_seconds()
    
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
    
    # 
    if wandb_run:
        # 
        wandb_run.log({
            "test_duration_seconds": test_duration,
            "test_samples": len(test_set),
            "test_batches": len(test_loader),
            "test_data_shape": f"{test_data.shape[0]}x{test_data.shape[1]}",
            "test_window_size": args.window_size,
            "test_horizon": args.horizon,
        })
        
        # 
        if 'mae_node' in performance_metrics:
            mae_node = performance_metrics['mae_node']
            mape_node = performance_metrics['mape_node'] 
            rmse_node = performance_metrics['rmse_node']
            
            # 
            wandb_run.log({
                "test_mae_node_mean": np.mean(mae_node),
                "test_mae_node_std": np.std(mae_node),
                "test_mae_node_min": np.min(mae_node),
                "test_mae_node_max": np.max(mae_node),
                "test_mape_node_mean": np.mean(mape_node),
                "test_mape_node_std": np.std(mape_node),
                "test_mape_node_min": np.min(mape_node),
                "test_mape_node_max": np.max(mape_node),
                "test_rmse_node_mean": np.mean(rmse_node),
                "test_rmse_node_std": np.std(rmse_node),
                "test_rmse_node_min": np.min(rmse_node),
                "test_rmse_node_max": np.max(rmse_node),
            })
        
        # 
        wandb_run.log({
            "test_batch_size": args.batch_size,
            "test_device": args.device,
            "test_norm_method": args.norm_method,
        })
        
        print("✓ wandb")
    
    return performance_metrics