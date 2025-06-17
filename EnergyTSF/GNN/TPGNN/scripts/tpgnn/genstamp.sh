data_path='datasets/V_France_processed_0.csv' #path to the MTS data ############
cycle=$((4*24)) #12 samples an hour, 24 hour a day ##############
data_root='datasets' #Directory to the MTS data
#preparing dataset stamp
python ./data_provider/data_process.py gen_stamp --data_path=$data_path --cycle=$cycle --data_root=$data_root


# bash ./scripts/tpgnn/genstamp.sh