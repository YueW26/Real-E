
'''

# Script to extract log details
input_path = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/tuning_results_G_192.log'
output_path = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/extracted_parameters.txt'

# Function to extract parameters
def extract_parameters(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    extracted_sections = []
    current_section = []
    recording = False
    
    for line in lines:
        if "====================" in line:
            if recording and current_section:
                extracted_sections.append(''.join(current_section))
                current_section = []
            recording = True
        elif "user config:" in line:
            if recording and current_section:
                extracted_sections.append(''.join(current_section))
                current_section = []
            recording = False
        if recording:
            current_section.append(line)
    
    # Save to output file
    with open(output_file, 'w') as f:
        for section in extracted_sections:
            f.write(section + '\n' + '=' * 20 + '\n')
    
    print(f"Extraction complete! Results saved to {output_file}")

# Run extraction
extract_parameters(input_path, output_path)

# EnergyTSF/TPGNN/log_details.py



'''



import csv
import re

input_file = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/extracted_parameters.txt'
output_file = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/TPGNN/extracted_results.csv'


def parse_logs_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Prepare data structure
    data_rows = []
    parameter_pattern = re.compile(r'Finished training with: (.*)')
    metric_start_pattern = re.compile(r'(MAE_mean|MAPE_mean|RMSE_mean): \[')
    metric_continue_pattern = re.compile(r'\s*(.+?)\s*(?:\]|,)$')  # Captures continued values across lines
    params_columns = ["lr", "bs", "nh", "ra", "dp", "wd", "ah", "ck", "kt"]
    
    current_params = None
    current_metric = None
    metric_values = []

    for i, line in enumerate(lines):
        # Match parameter line
        match_params = parameter_pattern.search(line)
        if match_params:
            # Save any ongoing metric before moving to new parameters
            if current_metric and metric_values:
                if len(metric_values) == 32:
                    row = list(current_params.values()) + [current_metric] + metric_values
                    data_rows.append(row)
                    print(f"[Debug] Added row for metric {current_metric}, values: {metric_values}")
                else:
                    print(f"[Warning] Metric {current_metric} has incomplete values: {metric_values}")
                metric_values = []

            params_str = match_params.group(1)
            params = {}
            for param in params_columns:
                match = re.search(fr'{param}([\d\.e\-]+)', params_str)
                params[param] = match.group(1) if match else None
            current_params = params
            print(f"[Debug] Extracted parameters: {current_params}")
            continue
        
        # Match start of a metric
        match_metric_start = metric_start_pattern.search(line)
        if match_metric_start:
            if current_metric and metric_values:
                # Save the previous metric data
                if len(metric_values) == 32:
                    row = list(current_params.values()) + [current_metric] + metric_values
                    data_rows.append(row)
                    print(f"[Debug] Added row for metric {current_metric}, values: {metric_values}")
                else:
                    print(f"[Warning] Metric {current_metric} has incomplete values: {metric_values}")
                metric_values = []

            current_metric = match_metric_start.group(1)
            values_part = line.split(': [', 1)[1].strip().rstrip(']')
            metric_values = values_part.split()
            print(f"[Debug] Start metric {current_metric}, initial values: {metric_values}")
            continue
        
        # Match continuation of a metric
        if current_metric:
            match_metric_continue = metric_continue_pattern.search(line)
            if match_metric_continue:
                values_part = match_metric_continue.group(1).strip()
                metric_values.extend(values_part.split())
                print(f"[Debug] Continuing metric {current_metric}, added values: {values_part}")
                if line.strip().endswith(']'):
                    if len(metric_values) == 32:
                        row = list(current_params.values()) + [current_metric] + metric_values
                        data_rows.append(row)
                        print(f"[Debug] Completed metric {current_metric}, final values: {metric_values}")
                    else:
                        print(f"[Warning] Metric {current_metric} has incomplete values: {metric_values}")
                    metric_values = []
                    current_metric = None

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(params_columns + ["Metric"] + [f"Value_{i+1}" for i in range(32)])
        # Write data rows
        writer.writerows(data_rows)

    print(f"Data successfully written to {output_file} with {len(data_rows)} rows.")

# Execute the function
parse_logs_to_csv(input_file, output_file)
