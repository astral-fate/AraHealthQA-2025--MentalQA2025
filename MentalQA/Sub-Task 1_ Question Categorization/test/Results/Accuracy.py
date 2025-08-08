import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
# TODO: Ensure these paths are correct for your Google Drive setup.
# Path to the directory containing all your model prediction .tsv files.
# Path to the directory containing all your model prediction .tsv files.
PREDICTIONS_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/results/'

# Path to the ground truth labels file.
# Based on your previous scripts, this seems to be the correct path.
GROUND_TRUTH_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/data/subtask1_output_test.tsv'

# Define the complete set of possible labels for the binarizer
ALL_LABELS = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'Z'])

# --- Helper Functions ---

def load_labels_from_file(filepath):
    """
    Loads labels from a single-column TSV file and splits them into lists.
    Handles potential file reading errors.
    """
    try:
        # Read the single column, treating all data as strings
        labels_df = pd.read_csv(filepath, sep='\t', header=None, names=['labels'], dtype=str, on_bad_lines='warn')
        # Handle potential empty rows or NaN values by filling with an empty string
        labels_df['labels'] = labels_df['labels'].fillna('')
        # Split the comma-separated strings into lists of labels
        labels_list = labels_df['labels'].apply(lambda s: [label.strip() for label in s.split(',') if label.strip()]).tolist()
        return labels_list
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

def main():
    """
    Main function to execute the EDA: load predictions, calculate F1 scores,
    and display a summary table. Skips corrupted files.
    """
    print("--- Starting Exploratory Data Analysis (EDA) for Model F1 Scores ---")

    # 1. Load and process ground truth labels
    print(f"Loading ground truth labels from: {GROUND_TRUTH_PATH}")
    true_labels_list = load_labels_from_file(GROUND_TRUTH_PATH)
    if true_labels_list is None:
        print("Halting execution because ground truth file could not be loaded.")
        return
    
    num_ground_truth_samples = len(true_labels_list)
    print(f"Ground truth loaded. Expected number of samples: {num_ground_truth_samples}")

    # 2. Binarize the ground truth labels for scoring
    mlb = MultiLabelBinarizer(classes=ALL_LABELS)
    y_true_binarized = mlb.fit_transform(true_labels_list)
    

    # 3. Iterate through prediction files, calculate scores, and store results
    results = []
    print(f"\nScanning for prediction files in: {PREDICTIONS_DIR}")

    if not os.path.exists(PREDICTIONS_DIR):
        print(f"FATAL: The directory '{PREDICTIONS_DIR}' does not exist. Please check the path.")
        return
        
    prediction_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.endswith('.tsv')]

    if not prediction_files:
        print("No .tsv prediction files found in the directory. Please check the contents.")
        return

    for filename in prediction_files:
        filepath = os.path.join(PREDICTIONS_DIR, filename)
        model_name = filename.replace('.tsv', '').replace('prediction_subtask_1_', '').replace('_test', '').replace('_', ' ').title()
        
        # Load prediction labels
        pred_labels_list = load_labels_from_file(filepath)
        if pred_labels_list is None:
            continue # Skip if file could not be read

        # --- ROBUSTNESS CHECK ---
        # Check for inconsistent number of samples before scoring
        if len(pred_labels_list) != num_ground_truth_samples:
            print(f"  - WARNING: Skipping '{model_name}'. Inconsistent number of samples. "
                  f"Expected {num_ground_truth_samples}, but found {len(pred_labels_list)}.")
            continue # Skip this corrupted file
        
        # Binarize prediction labels
        y_pred_binarized = mlb.transform(pred_labels_list)

        # Calculate weighted F1 score
        weighted_f1 = f1_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0)
        
        results.append({
            'Model Name': model_name,
            'Weighted F1-Score': weighted_f1
        })
        print(f"  - Calculated F1 for '{model_name}': {weighted_f1:.4f}")

    # 4. Display the final results in a sorted table
    if not results:
        print("\nNo valid results were calculated. Please check your prediction files.")
        return

    print("\n--- Final Model Performance Summary ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Weighted F1-Score', ascending=False).reset_index(drop=True)
    
    # Set display options for better table formatting
    pd.set_option('display.width', 100)
    pd.set_option('display.colheader_justify', 'left')

    print(results_df)
    print("---------------------------------------")
    print("\nEDA script finished successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly.
    # In a Colab notebook, you would just run the main() function in a cell.
    main()
