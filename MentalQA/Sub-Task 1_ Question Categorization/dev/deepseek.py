import os
import pandas as pd
from groq import Groq
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score
import time
import re

# --- Configuration ---
# IMPORTANT: Set your Groq API key in Google Colab's Secrets (named 'GROQ_API_KEY')
# or as a system environment variable.
GROQ_API_KEY = None
try:
    # First, try to get the key from Google Colab's secrets manager
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    print("Successfully loaded GROQ_API_KEY from Colab Secrets.")
except (ImportError, KeyError):
    # If not in Colab or key is not found, fall back to environment variable
    print("Could not load key from Colab Secrets. Checking for environment variable.")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if GROQ_API_KEY:
        print("Successfully loaded GROQ_API_KEY from environment variable.")


# Updated to use the DeepSeek model as requested
MODEL_NAME = 'deepseek-r1-distill-llama-70b'

# --- File Paths ---
# Corrected path from 'myDrive' to 'MyDrive'
DEV_DATA_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/dev_data.tsv'
DEV_LABELS_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/dev_label.tsv'
PREDICTION_OUTPUT_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/output/prediction_subtask_1.tsv'

# --- Helper Functions ---

def robust_read_lines(file_path):
    """
    Robustly reads a text file by trying different encodings.
    Returns a list of cleaned lines.
    """
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = [line.strip() for line in f.readlines()]
            print(f"Successfully read {file_path} with encoding '{enc}'.")
            return lines
        except (UnicodeDecodeError, TypeError):
            print(f"Failed to read {file_path} with encoding '{enc}'. Trying next...")
    print(f"FATAL: Could not read {file_path} with any of the attempted encodings.")
    return None

def load_and_prepare_data(data_path, labels_path):
    """
    Loads data and labels from two separate files, assuming a line-by-line correspondence.

    Args:
        data_path (str): Path to the file containing questions, one per line.
        labels_path (str): Path to the file containing labels, one per line.

    Returns:
        pandas.DataFrame: A merged DataFrame with 'ID', 'Question', and 'True_Labels' columns.
                          Returns None if files cannot be read or have mismatched lengths.
    """
    try:
        # Read questions and labels from their respective files
        questions = robust_read_lines(data_path)
        labels = robust_read_lines(labels_path)

        if questions is None or labels is None:
            return None

        # Check if the number of questions matches the number of labels
        if len(questions) != len(labels):
            print(f"FATAL: Mismatch in line count between data ({len(questions)} lines) and labels ({len(labels)} lines).")
            return None

        # Create a DataFrame assuming a line-by-line correspondence
        # An ID is generated based on the line number.
        merged_df = pd.DataFrame({
            'ID': range(len(questions)),
            'Question': questions,
            'True_Labels': labels
        })

        print("\nData loaded and combined successfully based on line order.")
        print(f"Total questions to process: {len(merged_df)}")
        print("\n--- Examining Combined Data ---")
        print(merged_df.head())
        print("-----------------------------\n")
        return merged_df

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the file paths are correct.")
        return None

def get_all_possible_labels(labels_series):
    """
    Extracts all unique labels from the labels column.

    Args:
        labels_series (pd.Series): A pandas Series containing comma-separated label strings.

    Returns:
        list: A sorted list of unique labels.
    """
    all_labels = set()
    for label_list in labels_series.dropna():
        for label in label_list.split(','):
            # Clean up whitespace from each label to ensure consistency
            cleaned_label = label.strip()
            if cleaned_label: # Avoid adding empty strings
                all_labels.add(cleaned_label)
    return sorted(list(all_labels))

def get_prediction_from_groq(client, question, all_labels):
    """
    Gets a multi-label classification prediction using a Chain-of-Thought prompt.

    Args:
        client (Groq): The initialized Groq API client.
        question (str): The patient question to classify.
        all_labels (list): The list of all possible labels for classification.

    Returns:
        str: A comma-separated string of predicted labels (e.g., "A,C").
    """
    # This Chain-of-Thought (CoT) prompt guides the model to reason before answering.
    prompt = f"""
    You are an expert AI specializing in analyzing questions related to mental health.
    Your task is to perform multi-label classification on the provided patient question.

    Follow these steps:
    1.  **Analyze the Question:** Briefly summarize the user's core concern in one sentence.
    2.  **Identify Themes:** Based on your analysis, list the key themes present in the question.
    3.  **Select Categories:** Compare the identified themes against the "Available Categories" list. Select all categories that are a strong match.
    4.  **Final Answer:** Provide your final answer as a comma-separated list of the selected category labels. This should be the last line of your response.

    **Available Categories:**
    {', '.join(all_labels)}

    **Example:**
    Question: "What are the long-term side effects of using Sertraline for anxiety?"

    **Your Thought Process:**
    1.  **Analyze the Question:** The user is asking about the potential negative effects of a specific medication used for anxiety.
    2.  **Identify Themes:** The key themes are medication, side effects, and anxiety.
    3.  **Select Categories:** "B" (Treatment, as it relates to medication) and "D" (Symptoms & Side Effects) are strong matches.
    4.  **Final Answer:** B, D

    ---
    **User's Question to Classify:**
    Question: "{question}"

    **Your Thought Process:**
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
            temperature=0.2, # Lower temperature for more deterministic reasoning
            max_tokens=1024,
            top_p=0.95,
        )
        response = chat_completion.choices[0].message.content.strip()

        # Extract the final answer, which is expected to be the last non-empty line.
        last_line = response.split('\n')[-1]

        # Clean the final line to get the predicted labels
        predicted_labels = [label.strip() for label in last_line.replace("Final Answer:", "").split(',') if label.strip() in all_labels]

        if not predicted_labels:
            print(f"Warning: Model did not return a valid label for a question. Response was: {response}")
            return "" # Return empty string if no valid labels found

        return ",".join(predicted_labels)

    except Exception as e:
        print(f"An API error occurred: {e}")
        return "" # Return empty string on error

def evaluate_predictions(true_labels, pred_labels, all_labels):
    """
    Calculates and prints the Weighted F1 and Jaccard scores.

    Args:
        true_labels (list): A list of true label strings.
        pred_labels (list): A list of predicted label strings.
        all_labels (list): The master list of all possible unique labels.
    """
    # Initialize the binarizer with all possible classes
    mlb = MultiLabelBinarizer(classes=all_labels)

    # Preprocess labels into lists of strings, stripping whitespace from each label.
    true_labels_split = [[label.strip() for label in str(s).split(',')] for s in true_labels]
    pred_labels_split = [[label.strip() for label in str(s).split(',')] for s in pred_labels]

    # Transform labels into binary matrix format
    y_true = mlb.fit_transform(true_labels_split)
    y_pred = mlb.transform(pred_labels_split)

    # Calculate scores
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Jaccard Score:     {jaccard:.4f}")
    print("--------------------------\n")

# --- Main Execution ---

def main():
    """
    Main function to run the entire classification and evaluation pipeline.
    """
    print("Starting MentalQA Sub-Task 1: Question Categorization...")

    if not GROQ_API_KEY:
        print("FATAL ERROR: GROQ_API_KEY not found in Colab Secrets or environment variables.")
        print("Please ensure the key is set correctly and try again.")
        return

    # 1. Load and Prepare Data
    dev_df = load_and_prepare_data(DEV_DATA_PATH, DEV_LABELS_PATH)
    if dev_df is None:
        return # Stop execution if data loading failed

    if dev_df.empty:
        print("\nWARNING: The loaded data is empty after merging.")
        print("Please check your input files. Halting execution.")
        return

    # 2. Initialize Groq Client
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        return

    # 3. Determine all possible labels
    all_labels = get_all_possible_labels(dev_df['True_Labels'])
    print(f"Discovered {len(all_labels)} unique labels: {all_labels}")

    # 4. Generate Predictions
    print(f"\nGenerating predictions for {len(dev_df)} questions using '{MODEL_NAME}' with Chain-of-Thought prompting...")
    predictions = []

    # Using tqdm for a progress bar
    for index, row in tqdm(dev_df.iterrows(), total=dev_df.shape[0], desc="Classifying Questions"):
        question_text = row['Question']
        prediction = get_prediction_from_groq(client, question_text, all_labels)
        predictions.append(prediction)

        # Add a small delay to respect API rate limits if necessary
        time.sleep(0.5)

    dev_df['Predicted_Labels'] = predictions
    print("Prediction generation complete.")

    # 5. Save Predictions to File
    output_dir = os.path.dirname(PREDICTION_OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(PREDICTION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for label in dev_df['Predicted_Labels']:
            f.write(f"{label}\n")
    print(f"Predictions saved to '{PREDICTION_OUTPUT_PATH}'")

    # 6. Evaluate the Results
    evaluate_predictions(
        true_labels=dev_df['True_Labels'].tolist(),
        pred_labels=dev_df['Predicted_Labels'].tolist(),
        all_labels=all_labels
    )

    print("Script finished successfully.")


if __name__ == "__main__":
    main()
