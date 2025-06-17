
#!/bin/bash

# Define directories and paths
DATA_ROOT='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/datasets'
STAMP_PATH="${DATA_ROOT}/time_stamp_G_192.npy"
MAIN_SCRIPT='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/TPGNN/main_tpgnn.py'

# Hyperparameter ranges
LEARNING_RATES=(1e-4)
BATCH_SIZES=(128)
N_HIDS=(128 256)
REG_AS=(1e-3 1e-4)
DROP_PROBS=(0.1 0.2)
WEIGHT_DECAYS=(1e-4 1e-5)
ATTN_HEADS=(1 2)
CE_KERNELS=(1 5)
STSTAMP_KTS=(2 4)

# Loop through all combinations
for LR in "${LEARNING_RATES[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for N_HID in "${N_HIDS[@]}"; do
      for REG_A in "${REG_AS[@]}"; do
        for DROP_PROB in "${DROP_PROBS[@]}"; do
          for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
            for ATTN_HEAD in "${ATTN_HEADS[@]}"; do
              for CE_KERNEL in "${CE_KERNELS[@]}"; do
                for KT in "${STSTAMP_KTS[@]}"; do
                  
                  # Construct a unique name for this experiment
                  EXP_NAME="lr${LR}_bs${BATCH_SIZE}_nh${N_HID}_ra${REG_A}_dp${DROP_PROB}_wd${WEIGHT_DECAY}_ah${ATTN_HEAD}_ck${CE_KERNEL}_kt${KT}"
                  
                  # Run the training script with the current hyperparameter set
                  python "$MAIN_SCRIPT" train \
                      --device=0 \
                      --n_route=16 \
                      --n_his=96 \
                      --n_pred=96 \
                      --n_train=34 \
                      --n_val=5 \
                      --n_test=5 \
                      --mode=1 \
                      --name="$EXP_NAME" \
                      --data_path="${DATA_ROOT}/V_Germany_processed_0.csv" \
                      --adj_matrix_path="${DATA_ROOT}/W_Germany_processed_0.csv" \
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
                      --STstamp.kt="$KT"
                  
                  # Optionally log the results
                  echo "Finished training with: $EXP_NAME"

                done
              done
            done
          done
        done
      done
    done
  done
done


# bash ./scripts/tpgnn/tune.sh
# bash ./scripts/tpgnn/tune.sh > tuning_results.log 2>&1
# bash ./scripts/tpgnn/tune.sh 2>&1 | tee tuning_results.log
# bash ./scripts/tpgnn/tune.sh 2>&1 | tee tuning_results_G_96_0512.log
