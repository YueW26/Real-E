#!/bin/bash

# Define directories and paths
DATA_ROOT='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/datasets'
STAMP_PATH="${DATA_ROOT}/time_stamp_F_96_R_0301.npy" #time_stamp_G_192.npy   time_stamp.npy
MAIN_SCRIPT='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/TPGNN/main_tpgnn.py'

# Hyperparameter ranges
LEARNING_RATES=(1e-4 1e-3)
BATCH_SIZES=(64)  #  128 256
N_HIDS=(64 128 256) 
REG_AS=(1e-3 1e-4 1e-5)
DROP_PROBS=(0.1 0.2 0.3)
WEIGHT_DECAYS=(1e-4 1e-5 1e-6)
ATTN_HEADS=(1 2 4)
CE_KERNELS=(1 3 5)_
STSTAMP_KTS=(2 4 6)

# Number of random samples to generate
NUM_SAMPLES=20

# Output CSV file
RESULTS_FILE="tuning_results_F_96_R_0301.csv" ###########################
echo "lr,batch_size,n_hid,reg_A,drop_prob,weight_decay,attn_head,ce_kernel,ststamp_kt,exp_name" > $RESULTS_FILE

# Random search
for ((i=1; i<=NUM_SAMPLES; i++)); do
  LR=${LEARNING_RATES[$((RANDOM % ${#LEARNING_RATES[@]}))]}
  BATCH_SIZE=${BATCH_SIZES[$((RANDOM % ${#BATCH_SIZES[@]}))]}
  N_HID=${N_HIDS[$((RANDOM % ${#N_HIDS[@]}))]}
  REG_A=${REG_AS[$((RANDOM % ${#REG_AS[@]}))]}
  DROP_PROB=${DROP_PROBS[$((RANDOM % ${#DROP_PROBS[@]}))]}
  WEIGHT_DECAY=${WEIGHT_DECAYS[$((RANDOM % ${#WEIGHT_DECAYS[@]}))]}
  ATTN_HEAD=${ATTN_HEADS[$((RANDOM % ${#ATTN_HEADS[@]}))]}
  CE_KERNEL=${CE_KERNELS[$((RANDOM % ${#CE_KERNELS[@]}))]}
  STSTAMP_KT=${STSTAMP_KTS[$((RANDOM % ${#STSTAMP_KTS[@]}))]}

  # Construct a unique name for this experiment
  EXP_NAME="lr${LR}_bs${BATCH_SIZE}_nh${N_HID}_ra${REG_A}_dp${DROP_PROB}_wd${WEIGHT_DECAY}_ah${ATTN_HEAD}_ck${CE_KERNEL}_kt${STSTAMP_KT}"

  # Run the training script with the current hyperparameter set
  python "$MAIN_SCRIPT" train \
      --device=0 \
      --n_route=10 \
      --n_his=96 \
      --n_pred=96 \
      --n_train=34 \
      --n_val=5 \
      --n_test=5 \
      --mode=1 \
      --name="$EXP_NAME" \
      --data_path="${DATA_ROOT}/V_France_processed_0.csv" \
      --adj_matrix_path="${DATA_ROOT}/F_granger_causality_matri.csv" \
      --seq_in_len=96 \
      --seq_out_len=96 \
      --stamp_path=$STAMP_PATH \
      --epochs=100 \
      --lr="$LR" \
      --batch_size="$BATCH_SIZE" \
      --n_hid="$N_HID" \
      --reg_A="$REG_A" \
      --drop_prob="$DROP_PROB" \
      --adam.weight_decay="$WEIGHT_DECAY" \
      --attn.head="$ATTN_HEAD" \
      --CE.kernel_size="$CE_KERNEL" \
      --STstamp.kt="$STSTAMP_KT" \
      --day_slot=576

  # Log the results in CSV format
  echo "$LR,$BATCH_SIZE,$N_HID,$REG_A,$DROP_PROB,$WEIGHT_DECAY,$ATTN_HEAD,$CE_KERNEL,$STSTAMP_KT,$EXP_NAME" >> $RESULTS_FILE

  # Optionally log the progress
  echo "Finished training with: $EXP_NAME"
done

# Optional: notify when the script is complete
echo "Random search tuning completed. Results saved to $RESULTS_FILE."


# bash ./scripts/tpgnn/tune_r.sh 2>&1 | tee turn_r_result_F_96_0301_W2_.log





'''
#!/bin/bash

# Define directories and paths
DATA_ROOT='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/datasets'
STAMP_PATH="${DATA_ROOT}/time_stamp_F_96.npy" #time_stamp_G_192.npy   time_stamp.npy
MAIN_SCRIPT='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/TPGNN/main_tpgnn.py'

# Hyperparameter ranges
LEARNING_RATES=(1e-4 1e-3)
BATCH_SIZES=(64)  #  128 256
N_HIDS=(64 128 256) 
REG_AS=(1e-3 1e-4 1e-5)
DROP_PROBS=(0.1 0.2 0.3)
WEIGHT_DECAYS=(1e-4 1e-5 1e-6)
ATTN_HEADS=(1 2 4)
CE_KERNELS=(1 3 5)
STSTAMP_KTS=(2 4 6)

# Number of random samples to generate
NUM_SAMPLES=20

# Output CSV file
RESULTS_FILE="tuning_results_G_96_R_1312.csv" ###########################
echo "lr,batch_size,n_hid,reg_A,drop_prob,weight_decay,attn_head,ce_kernel,ststamp_kt,exp_name" > $RESULTS_FILE

# Random search
for ((i=1; i<=NUM_SAMPLES; i++)); do
  LR=${LEARNING_RATES[$((RANDOM % ${#LEARNING_RATES[@]}))]}
  BATCH_SIZE=${BATCH_SIZES[$((RANDOM % ${#BATCH_SIZES[@]}))]}
  N_HID=${N_HIDS[$((RANDOM % ${#N_HIDS[@]}))]}
  REG_A=${REG_AS[$((RANDOM % ${#REG_AS[@]}))]}
  DROP_PROB=${DROP_PROBS[$((RANDOM % ${#DROP_PROBS[@]}))]}
  WEIGHT_DECAY=${WEIGHT_DECAYS[$((RANDOM % ${#WEIGHT_DECAYS[@]}))]}
  ATTN_HEAD=${ATTN_HEADS[$((RANDOM % ${#ATTN_HEADS[@]}))]}
  CE_KERNEL=${CE_KERNELS[$((RANDOM % ${#CE_KERNELS[@]}))]}
  STSTAMP_KT=${STSTAMP_KTS[$((RANDOM % ${#STSTAMP_KTS[@]}))]}

  # Construct a unique name for this experiment
  EXP_NAME="lr${LR}_bs${BATCH_SIZE}_nh${N_HID}_ra${REG_A}_dp${DROP_PROB}_wd${WEIGHT_DECAY}_ah${ATTN_HEAD}_ck${CE_KERNEL}_kt${STSTAMP_KT}"

  # Run the training script with the current hyperparameter set
  python "$MAIN_SCRIPT" train \
      --device=0 \
      --n_route=10 \
      --n_his=96 \
      --n_pred=96 \
      --n_train=34 \
      --n_val=5 \
      --n_test=5 \
      --mode=1 \
      --name="$EXP_NAME" \
      --data_path="${DATA_ROOT}/V_France_processed_0.csv" \
      --adj_matrix_path="${DATA_ROOT}/W_France_processed_0.csv" \
      --seq_in_len=96 \
      --seq_out_len=96 \
      --stamp_path=$STAMP_PATH \
      --epochs=100 \
      --lr="$LR" \
      --batch_size="$BATCH_SIZE" \
      --n_hid="$N_HID" \
      --reg_A="$REG_A" \
      --drop_prob="$DROP_PROB" \
      --adam.weight_decay="$WEIGHT_DECAY" \
      --attn.head="$ATTN_HEAD" \
      --CE.kernel_size="$CE_KERNEL" \
      --STstamp.kt="$STSTAMP_KT"

  # Log the results in CSV format
  echo "$LR,$BATCH_SIZE,$N_HID,$REG_A,$DROP_PROB,$WEIGHT_DECAY,$ATTN_HEAD,$CE_KERNEL,$STSTAMP_KT,$EXP_NAME" >> $RESULTS_FILE

  # Optionally log the progress
  echo "Finished training with: $EXP_NAME"
done

# Optional: notify when the script is complete
echo "Random search tuning completed. Results saved to $RESULTS_FILE."


# bash ./scripts/tpgnn/tune_r.sh 2>&1 | tee turn_r_result_F_96_1312.log

'''