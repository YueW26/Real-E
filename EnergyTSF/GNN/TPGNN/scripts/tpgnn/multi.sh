
#!/usr/bin/env bash
# data_path='/home/kit/aifb/cc7738/scratch/EnergyTSF/datasets/V_228.csv' #path to the MTS data
# cycle=$((12*24)) #12 samples an hour, 24 hour a day
# data_root='/home/kit/aifb/cc7738/scratch/EnergyTSF/datasets' #Directory to the MTS data
# #preparing dataset stamp
# python /home/kit/aifb/cc7738/scratch/EnergyTSF/data_provider/data_process.py gen_stamp --data_path=$data_path --cycle=$cycle --data_root=$data_root

# data_path='/home/kit/aifb/cc7738/scratch/EnergyTSF/datasets/V_228.csv' #path to the MTS data
# adj_path='datasets/PeMS/W_228.csv'  #path to the adjacency matrix, None if not exists
data_root='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF_German/datasets' #Directory to the MTS data
stamp_path="${data_root}/time_stamp.npy"
#training model
python /mnt/webscistorage/cc7738/ws_joella/EnergyTSF_German/TPGNN/main_tpgnn.py train --device=0 --n_route=10 --n_his=96 --n_pred=96 --n_train=34 --n_val=5 --n_test=5 --mode=1 --name='PeMS'\
    --data_path="${data_root}/V_France_processed_0.csv" --adj_matrix_path="${data_root}/W_France_processed_0.csv" --seq_in_len=12 --seq_out_len=12 --stamp_path=$stamp_path --epochs=100




# /Users/wangbo/EnergyTSF-2/datasets/ --device cpu --data custom --task_name forecasting --data_path  Merged_Data_germany.csv
# /Users/wangbo/EnergyTSF-2/scripts/tpgnn/multi.sh
# chmod +x multi.sh
# ./multi.sh
# /Users/wangbo/EnergyTSF-2/scripts/tpgnn/multi.sh

# sh ./scripts/tpgnn/multi.sh





