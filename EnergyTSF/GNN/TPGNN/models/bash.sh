#!/bin/bash

# Define the commands in an array
#!/bin/bash

# Parameters to vary
window_sizes=(12 24)           # Values for window_size
devices=("cuda:0")       # Values for device

# Other fixed parameters
dataset="France_processed_0"
horizon=(3 9 15)
norm_method="z_score"
train_length=7
valid_length=2
test_length=(1 6 12)
root_path="/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/"
data="Opennem"
task_name="forecasting"
data_path="France_processed_0.csv"
target="'Fossil Gas  - Actual Aggregated [MW]'"

# Iterate over parameter combinations
for window_size in "${window_sizes[@]}"; do
    for test_l in "${test_length[@]}"; do
        for hor in "${horizon[@]}"; do
            # cmd="python Stemgnn/stem_gnn.py --train True --evaluate True --dataset $dataset --window_size $window_size --horizon $hor --norm_method $norm_method --train_length $test_l &
            echo "python Stemgnn/stem_gnn.py --train True --evaluate True --dataset $dataset --window_size $window_size --horizon $hor --norm_method $norm_method --train_length $test_l &"

# for devide_id, exp_id in zip(device_list, exp_list):
    python test.py --device $id exp_id $ exp_id

# python test.py --device 0 exp_id 0 
# python test.py --device 1 exp_id 1 
# python test.py --device 2 exp_id 2 
# python test.py --device 3 exp_id 3 

"""
#!/bin/bash

# Define the lists
device_list=(0 1 2 3) # List of device IDs
exp_list=(0 1 2 3)    # List of experiment IDs

# Get the length of the lists
len=${#device_list[@]}

# Iterate over the indices
for ((i=0; i<$len; i++)); do
    device_id=${device_list[$i]}
    exp_id=${exp_list[$i]}

    # Construct and run the command
    cmd="python test.py --device $device_id --exp_id $exp_id"
    echo "Running command: $cmd"
    $cmd & # Run in the background
done

# Wait for all background processes to finish
wait

echo "All commands have finished executing."

"""