import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score
import re
import time # Import the time module

# --- Configuration ---
# IMPORTANT: Set your NVIDIA API key in Google Colab's Secrets (named 'NVIDIA_API_KEY')
# or as a system environment variable.
NVIDIA_API_KEY = None
try:
    from google.colab import userdata
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
    print("Successfully loaded NVIDIA_API_KEY from Colab Secrets.")
except (ImportError, KeyError):
    print("Could not load key from Colab Secrets. Checking for environment variable.")
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
    if NVIDIA_API_KEY:
        print("Successfully loaded NVIDIA_API_KEY from environment variable.")

# --- API and Model Configuration ---
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "writer/palmyra-med-70b-32k"

# --- File Paths ---
TEST_DATA_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/data/subtask1_input_test.tsv'
TEST_LABELS_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/data/subtask1_output_test.tsv'
OUTPUT_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/results/' # Changed to match your results folder


# --- Helper Functions ---
def robust_read_lines(file_path):
    """Reads a text file, trying different encodings to avoid errors."""
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')[0].astype(str).tolist()
            print(f"Successfully read {file_path} with encoding '{enc}'.")
            return [line.strip() for line in lines]
        except Exception:
            continue
    print(f"FATAL: Could not read {file_path} with any of the attempted encodings.")
    return None

def load_and_prepare_data(data_path, labels_path):
    """Loads test questions and their corresponding ground truth labels and merges them."""
    try:
        questions = robust_read_lines(data_path)
        labels = robust_read_lines(labels_path)

        if questions is None or labels is None:
            return None
        if len(questions) != len(labels):
            print(f"FATAL: Mismatch in line count between questions ({len(questions)}) and labels ({len(labels)}).")
            return None

        merged_df = pd.DataFrame({'Question': questions, 'True_Labels': labels})
        print(f"\nData loaded successfully. Total questions to process: {len(merged_df)}")
        return merged_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the file paths are correct and the Drive is mounted.")
        return None

def build_prompt(question):
    """Builds the advanced prompt for the model."""
    prompt = f"""
You are an expert in classifying Arabic patient questions into mental health categories. Perform precise multi-label classification.

**Categories:**
- (A) Diagnosis: Interpreting symptoms.
- (B) Treatment: Seeking therapies or medications.
- (C) Anatomy and Physiology: Basic medical knowledge.
- (D) Epidemiology: Course, prognosis, causes of diseases.
- (E) Healthy Lifestyle: Diet, exercise, mood control.
- (F) Provider Choices: Recommendations for doctors.
- (Z) Other: Does not fit other categories.

**Example 1:**
Question: "هل يعتبر الخوف من عدم الإنجاب مستقبلاً حالة عادية خاصةً لما أكون متعلقة بأطفال كثيراً وأنا على وجه جواز أنا خايفة جداً"
Reasoning: The user is asking if their fear (a symptom) is normal and is concerned about its future course (prognosis). This fits 'Diagnosis' (interpreting a symptom) and 'Epidemiology' (prognosis).
Final Answer: A,D

**Example 2:**
Question: "من سنه تقريبا و انا أذي نفسي ب اكثر من طريقة و ما اعرف كيف اتخلص من ذي العادة، و بدت تجيني افكار بإنهاء حياتي و حاولت انتحر باكثر من مرة و أكثر من طريقة"
Reasoning: The user describes self-harm and suicidal thoughts and is asking how to get rid of this habit. This is a clear call for 'Treatment' (seeking therapy/help) and relates to 'Healthy Lifestyle' (self-help, mood control).
Final Answer: B,E

**Your Task:**
Classify the following question. Provide your reasoning and then the final answer.

**Question:** "{question}"

**Reasoning:**
Final Answer:"""
    return prompt

def get_prediction_from_nvidia_api(client, question, all_labels, model_name):
    """Gets a multi-label prediction from the NVIDIA API."""
    full_prompt = build_prompt(question)
    full_response_content = ""
    
    try:
        completion = client.chat.completions.create(
          model=model_name,
          messages=[{"role":"user", "content": full_prompt}],
          temperature=0.2, top_p=0.7, max_tokens=1024, stream=True
        )
        for chunk in completion:
          if chunk.choices[0].delta.content is not None:
            full_response_content += chunk.choices[0].delta.content

        match = re.search(r"Final Answer:\s*([A-Z,]+)", full_response_content, re.IGNORECASE)
        if match:
            predicted_labels_str = match.group(1)
            predicted_labels = [label.strip().upper() for label in predicted_labels_str.split(',') if label.strip().upper() in all_labels]
            return ",".join(sorted([label for label in predicted_labels if label]))
        else:
            print(f"Warning: Model '{model_name}' did not return a valid, formatted label. Response: '{full_response_content}'")
            return ""
    except Exception as e:
        print(f"An API error occurred for model {model_name}: {e}")
        return ""

def evaluate_predictions(true_labels_str_list, pred_labels_str_list, all_labels):
    """Calculates and prints the Weighted F1 and Jaccard scores."""
    mlb = MultiLabelBinarizer(classes=all_labels)
    
    true_labels_split = [[label.strip() for label in s.split(',')] for s in true_labels_str_list]
    pred_labels_split = [[label.strip() for label in s.split(',')] for s in pred_labels_str_list]

    y_true = mlb.fit_transform(true_labels_split)
    y_pred = mlb.transform(pred_labels_split)

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- Final Evaluation Results ---")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Jaccard Score:     {jaccard:.4f}")
    print("--------------------------------\n")

# --- Main Execution ---
def main():
    """Main function to run the classification, timing, saving, and evaluation process."""
    print(f"🚀 Starting LLM-based Question Categorization using '{MODEL_NAME}'...")
    if not NVIDIA_API_KEY:
        print("❌ FATAL: NVIDIA_API_KEY not found. Please set it in Colab Secrets or as an environment variable.")
        return

    test_df = load_and_prepare_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    if test_df is None or test_df.empty:
        print("❌ Halting execution due to data loading issues.")
        return

    try:
        client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
        print("✅ NVIDIA API client initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize NVIDIA API client: {e}")
        return

    all_labels = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'Z'])
    print(f"ℹ️ Using predefined labels for classification: {all_labels}")

    print(f"\n⏳ Generating predictions for {len(test_df)} test questions...")
    
    # --- ADDED: Start timer ---
    start_time = time.time()
    
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc=f"Classifying with {MODEL_NAME.split('/')[0]}"):
        prediction = get_prediction_from_nvidia_api(client, row['Question'], all_labels, MODEL_NAME)
        predictions.append(prediction)
    
    # --- ADDED: End timer and calculate duration ---
    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    
    # --- MODIFIED: Ensure the new predictions are in the DataFrame ---
    test_df['Predicted_Labels'] = predictions
    print("✅ Prediction generation complete.")

    # --- Save the CORRECT predictions ---
    safe_model_name = MODEL_NAME.replace('/', '_')
    # MODIFIED: Changed output directory to 'results' to match your EDA script
    prediction_output_path = os.path.join(OUTPUT_DIR, f"{safe_model_name}.tsv") 

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✅ Created output directory: {OUTPUT_DIR}")

    # Use the 'Predicted_Labels' column to save the file
    test_df[['Predicted_Labels']].to_csv(prediction_output_path, sep='\t', header=False, index=False)
    print(f"💾 Predictions correctly saved to '{prediction_output_path}'")

    # --- Evaluate the new predictions ---
    evaluate_predictions(
        test_df['True_Labels'].tolist(),
        test_df['Predicted_Labels'].tolist(), # Use the generated predictions
        all_labels
    )
    
    # --- ADDED: Print total time taken ---
    print(f"⏱️ Total time taken for prediction: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes).")
    print("\n\n✅ Script finished successfully.")


if __name__ == "__main__":
    main()
