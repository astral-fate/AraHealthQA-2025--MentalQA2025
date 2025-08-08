import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score
import re

# --- Configuration ---
# IMPORTANT: Set your NVIDIA API key in Google Colab's Secrets (named 'NVIDIA_API_KEY')
# or as a system environment variable.
NVIDIA_API_KEY = None
try:
    # Best practice for securely handling API keys in Colab
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
MODEL_NAME = "mistralai/mixtral-8x22b-instruct-v0.1"

# --- File Paths ---
TEST_DATA_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/data/subtask1_input_test.tsv'
TEST_LABELS_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/data/subtask1_output_test.tsv'
OUTPUT_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/output/'


# --- Helper Functions ---
def robust_read_lines(file_path):
    """Reads a text file, trying different encodings to avoid errors."""
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                # Using pd.read_csv for robust TSV parsing
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
            print("FATAL: Failed to read questions or labels files.")
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
    """
    Builds the advanced prompt with detailed instructions, category definitions,
    and few-shot examples to guide the model.
    """
    prompt = f"""
You are a world-class AI expert in classifying Arabic patient questions into predefined mental health categories. Your task is to perform precise multi-label classification.

**STEP 1: Understand the Categories**
First, carefully review the definitions of each available category:
- **(A) Diagnosis:** Questions about interpreting symptoms, conditions, or clinical findings.
- **(B) Treatment:** Questions about seeking or evaluating treatments, therapies, or medications.
- **(C) Anatomy and Physiology:** Questions about basic medical knowledge, how the body/mind works.
- **(D) Epidemiology:** Questions about the course, prognosis, and causes of diseases.
- **(E) Healthy Lifestyle:** Questions related to diet, exercise, mood control, and self-help.
- **(F) Provider Choices:** Questions seeking recommendations for doctors or facilities.
- **(Z) Other:** Any question that does not fit into the categories above.

**STEP 2: Learn from Examples (Few-Shot Learning)**
Here are some examples of how to correctly classify questions. Study their logic.

---
**Example 1:**
Question: "هل يعتبر الخوف من عدم الإنجاب مستقبلاً حالة عادية خاصةً لما أكون متعلقة بأطفال كثيراً وأنا على وجه جواز أنا خايفة جداً"
Reasoning: The user is asking if their fear (a symptom) is normal and is concerned about its future course (prognosis). This fits 'Diagnosis' (interpreting a symptom) and 'Epidemiology' (prognosis).
Final Answer: A,D

---
**Example 2:**
Question: "من سنه تقريبا و انا أذي نفسي ب اكثر من طريقة و ما اعرف كيف اتخلص من ذي العادة، و بدت تجيني افكار بإنهاء حياتي و حاولت انتحر باكثر من مرة و أكثر من طريقة"
Reasoning: The user describes self-harm and suicidal thoughts and is asking how to get rid of this habit. This is a clear call for 'Treatment' (seeking therapy/help) and relates to 'Healthy Lifestyle' (self-help, mood control).
Final Answer: B,E
---

**STEP 3: Classify the New Question (Your Task)**
Now, apply the same logic to the following question. Provide a step-by-step thought process and conclude with the 'Final Answer' on a new line.

**Question:** "{question}"

**Reasoning:**
Final Answer:"""
    return prompt

def get_prediction_from_nvidia_api(client, question, all_labels, model_name):
    """Gets a multi-label prediction using the NVIDIA API with streaming."""
    full_prompt = build_prompt(question)
    full_response_content = ""
    
    try:
        # API call with the specified model name and updated parameters
        completion = client.chat.completions.create(
          model=model_name,
          messages=[{"role":"user", "content": full_prompt}],
          temperature=0.5,
          top_p=1,
          max_tokens=1024,
          stream=True
        )

        # Process the stream to build the full response
        for chunk in completion:
          if chunk.choices[0].delta.content is not None:
            full_response_content += chunk.choices[0].delta.content

        # Once the full response is assembled, parse it
        match = re.search(r"Final Answer:\s*([A-Z,]+)", full_response_content, re.IGNORECASE)

        if match:
            predicted_labels_str = match.group(1)
            predicted_labels = [label.strip().upper() for label in predicted_labels_str.split(',') if label.strip().upper() in all_labels]
            predicted_labels = [label for label in predicted_labels if label]
        else:
            predicted_labels = []

        if not predicted_labels:
            print(f"Warning: Model '{model_name}' did not return a valid, formatted label. Response: '{full_response_content}'")
            return ""

        return ",".join(sorted(predicted_labels))

    except Exception as e:
        print(f"An API error occurred for model {model_name}: {e}")
        return ""

def evaluate_predictions(true_labels, pred_labels, all_labels):
    """Calculates and prints the Weighted F1 and Jaccard scores."""
    true_labels_str = [str(s) for s in true_labels]
    pred_labels_str = [str(s) for s in pred_labels]

    mlb = MultiLabelBinarizer(classes=all_labels)
    
    true_labels_split = [[label.strip() for label in s.split(',')] for s in true_labels_str]
    pred_labels_split = [[label.strip() for label in s.split(',')] for s in pred_labels_str]

    y_true = mlb.fit_transform(true_labels_split)
    y_pred = mlb.transform(pred_labels_split)

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Jaccard Score:     {jaccard:.4f}")
    print("--------------------------\n")

# --- Main Execution ---
def main():
    """Main function to run the classification and evaluation process."""
    print(f"Starting LLM-based Question Categorization using '{MODEL_NAME}'...")
    if not NVIDIA_API_KEY:
        print("FATAL: NVIDIA_API_KEY not found. Please set it in Colab Secrets or as an environment variable.")
        return

    test_df = load_and_prepare_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    if test_df is None or test_df.empty:
        print("Halting execution due to data loading issues.")
        return

    try:
        client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
        print("NVIDIA API client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize NVIDIA API client: {e}")
        return

    all_labels = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'Z'])
    print(f"Using predefined labels for classification: {all_labels}")

    print(f"\nGenerating predictions for {len(test_df)} questions...")
    predictions = []
    # Use tqdm for a progress bar
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc=f"Classifying with {MODEL_NAME.split('/')[0]}"):
        prediction = get_prediction_from_nvidia_api(client, row['Question'], all_labels, MODEL_NAME)
        predictions.append(prediction)
    
    test_df['Predicted_Labels'] = predictions
    print("Prediction generation complete.")

    # --- Save and Evaluate ---
    safe_model_name = MODEL_NAME.replace('/', '_')
    prediction_output_path = os.path.join(OUTPUT_DIR, f"prediction_subtask_1_{safe_model_name}_test.tsv")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    with open(prediction_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(predictions))
    print(f"Predictions saved to '{prediction_output_path}'")

    print(f"\n--- Evaluation Results for {MODEL_NAME} ---")
    evaluate_predictions(
        test_df['True_Labels'].tolist(),
        predictions,
        all_labels
    )

    print("\n\n✅ Script finished successfully.")


if __name__ == "__main__":
    main()
