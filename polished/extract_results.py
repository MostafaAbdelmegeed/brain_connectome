import re
import pandas as pd
import sys
import os
import numpy as np


def extract_info_from_log(file_path):
    data = {
        "Fold": [],
        "Best Validation Loss": [],
        "Best Accuracy": [],
        "Best Precision": [],
        "Best F1 Score": [],
        "Final Accuracy": None,
        "Final Precision": None,
        "Final F1 Score": None,
        "Final Confusion Matrix": None,
    }
    
    with open(file_path, 'r') as log_file:
        lines = list(log_file)  # Convert the file object into a list for safe iteration

    # Regular expressions to capture important information
    fold_re = re.compile(r"Fold (\d+)/\d+")
    best_loss_re = re.compile(r"Fold \d+ Best Validation Loss: ([\d.]+)")
    best_metrics_re = re.compile(r"Fold \d+ Best Metrics: Accuracy: ([\d.]+), Precision: ([\d.]+), F1 Score: ([\d.]+)")
    final_metrics_re = re.compile(r"Final Metrics \| Accuracy: ([\d.]+) \| Precision: ([\d.]+) \| F1 Score: ([\d.]+)")
    final_conf_matrix_re = re.compile(r"Final Confusion Matrix:")

    fold = None
    final_metrics_found = False

    for idx, line in enumerate(lines):
        # Match fold info
        fold_match = fold_re.search(line)
        if fold_match:
            fold = fold_match.group(1)
            continue
        
        # Match best validation loss
        best_loss_match = best_loss_re.search(line)
        if best_loss_match and fold:
            data["Fold"].append(fold)
            data["Best Validation Loss"].append(float(best_loss_match.group(1)))
            continue
        
        # Match best metrics
        best_metrics_match = best_metrics_re.search(line)
        if best_metrics_match and fold:
            data["Best Accuracy"].append(float(best_metrics_match.group(1)))
            data["Best Precision"].append(float(best_metrics_match.group(2)))
            data["Best F1 Score"].append(float(best_metrics_match.group(3)))
            continue
        
        # Match final metrics
        final_metrics_match = final_metrics_re.search(line)
        if final_metrics_match:
            data["Final Accuracy"] = float(final_metrics_match.group(1))
            data["Final Precision"] = float(final_metrics_match.group(2))
            data["Final F1 Score"] = float(final_metrics_match.group(3))
            final_metrics_found = True
            continue

        # Match final confusion matrix if "Final Metrics" was found
        if final_metrics_found and final_conf_matrix_re.search(line):
            # Next lines will have the confusion matrix, check for index safety
            confusion_matrix = []
            for i in range(4):
                if idx + i + 1 < len(lines):
                    matrix_line = lines[idx + i + 1].strip()
                    # Clean up matrix line by removing any brackets or extra characters
                    matrix_line_clean = matrix_line.replace("[", "").replace("]", "").strip()
                    try:
                        confusion_matrix.append([float(x) for x in matrix_line_clean.split()])
                    except ValueError:
                        # In case there's an issue converting the line, skip this matrix
                        print(f"Warning: Skipping invalid matrix line in file {file_path}: {matrix_line_clean}")
                        break
            if len(confusion_matrix) == 4:  # Ensure we have a complete confusion matrix
                data["Final Confusion Matrix"] = confusion_matrix

    if final_metrics_found:
        return data
    else:
        return None  # Indicates that the file did not contain final metrics

def save_to_excel(data, output_path):
    # DataFrame for Fold-wise metrics
    df_folds = pd.DataFrame({
        "Fold": data["Fold"],
        "Best Validation Loss": data["Best Validation Loss"],
        "Best Accuracy": data["Best Accuracy"],
        "Best Precision": data["Best Precision"],
        "Best F1 Score": data["Best F1 Score"]
    })

    # Calculate average, min, max, and std deviation (sample)
    stats = {
        "Metric": ["Average", "Min", "Max", "Std Dev (Sample)"],
        "Best Validation Loss": [
            np.mean(data["Best Validation Loss"]),
            np.min(data["Best Validation Loss"]),
            np.max(data["Best Validation Loss"]),
            np.std(data["Best Validation Loss"], ddof=1)  # Sample std dev
        ],
        "Best Accuracy": [
            np.mean(data["Best Accuracy"]),
            np.min(data["Best Accuracy"]),
            np.max(data["Best Accuracy"]),
            np.std(data["Best Accuracy"], ddof=1)
        ],
        "Best Precision": [
            np.mean(data["Best Precision"]),
            np.min(data["Best Precision"]),
            np.max(data["Best Precision"]),
            np.std(data["Best Precision"], ddof=1)
        ],
        "Best F1 Score": [
            np.mean(data["Best F1 Score"]),
            np.min(data["Best F1 Score"]),
            np.max(data["Best F1 Score"]),
            np.std(data["Best F1 Score"], ddof=1)
        ]
    }

    # DataFrame for calculated statistics
    df_stats = pd.DataFrame(stats)

    with pd.ExcelWriter(output_path) as writer:
        # Write Fold-wise results to the first sheet
        df_folds.to_excel(writer, sheet_name="Fold Results", index=False)
        
        # Append the calculated statistics
        df_stats.to_excel(writer, sheet_name="Fold Results", index=False, startrow=len(df_folds) + 2)
        
        # Final metrics and confusion matrix
        df_final_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "F1 Score"],
            "Value": [data["Final Accuracy"], data["Final Precision"], data["Final F1 Score"]]
        })
        df_final_metrics.to_excel(writer, sheet_name="Final Metrics", index=False)

        if data["Final Confusion Matrix"]:
            df_conf_matrix = pd.DataFrame(data["Final Confusion Matrix"], 
                                          columns=["Class 0", "Class 1", "Class 2", "Class 3"])
            df_conf_matrix.to_excel(writer, sheet_name="Final Confusion Matrix", index=False)

def rename_file(log_file_name):
    # Default to no prefix
    prefix = ""

    # Check for each condition and assign the appropriate prefix
    if "BrainNetGIN_BASELINE" in log_file_name:
        if any(x in log_file_name for x in ["_BASE_", "_corr_"]):
            prefix = "Baseline"
        elif "ASYM" in log_file_name:
            prefix = "Asymmetry"
    elif "BrainNetGIN" in log_file_name:
        if any(x in log_file_name for x in ["_BASE_", "_corr_", "_fenc_"]):
            prefix = "Encoded"
        elif "ASYM" in log_file_name:
            prefix = "Immersed"

    # Extract seed (s#)
    seed_match = re.search(r"s(\d+)", log_file_name)
    seed = f"s{seed_match.group(1)}" if seed_match else ""

    # Extract model type (BASE, ASYM, etc.)
    model_type_match = re.search(r"BASE|ASYM|fenc|corr", log_file_name)
    model_type = model_type_match.group(0) if model_type_match else ""

    # Extract dataset (adni, adni_multi, ppmi)
    dataset_match = re.search(r"adni_multi|adni|ppmi", log_file_name)
    dataset = dataset_match.group(0) if dataset_match else ""

    # Versioning logic for adni_multi
    version = ""
    if "adni_multi" in log_file_name:
        version = "_2" if "mine" in log_file_name else "_1"

    # Create the new file name based on the pattern
    new_name = f"{dataset}{version}_{prefix}_{seed}.xlsx"
    return new_name

def process_log_file(log_file_path, output_directory):
    log_file_name = os.path.basename(log_file_path)
    
    # Filter out files that don't meet the conditions
    if "ah" in log_file_name or "h1024" not in log_file_name or not re.search(r"d[\d\.]+", log_file_name):
        print(f"Skipping {log_file_name}: Doesn't meet filter criteria.")
        return

    # Rename the file using meaningful names
    new_file_name = rename_file(log_file_name)
    if new_file_name is None:
        print(f"Skipping {log_file_name}: Could not parse filename.")
        return
    
    excel_file_path = os.path.join(output_directory, new_file_name)

    extracted_data = extract_info_from_log(log_file_path)
    if extracted_data:
        save_to_excel(extracted_data, excel_file_path)
        print(f"Processed {log_file_name}, summary exported to {new_file_name}")
    else:
        print(f"Skipping {log_file_name}, no final metrics found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file_or_directory> [output_directory]")
        sys.exit(1)

    input_path = sys.argv[1]

    # If an output directory is provided, use it; otherwise, save in the current directory
    if len(sys.argv) == 3:
        output_directory = sys.argv[2]
    else:
        output_directory = "."

    # Check if input_path is a directory
    if os.path.isdir(input_path):
        # Process all .log files in the directory
        for file_name in os.listdir(input_path):
            if file_name.endswith(".log"):
                log_file_path = os.path.join(input_path, file_name)
                process_log_file(log_file_path, output_directory)
    else:
        # Process single log file
        if input_path.endswith(".log"):
            process_log_file(input_path, output_directory)
        else:
            print("Please provide a valid .log file or directory containing .log files.")
