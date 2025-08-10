import os
import pandas as pd
from groq import Groq
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score
import re

# --- Configuration ---
# IMPORTANT: Set your Groq API key in Google Colab's Secrets (named 'GROQ_API_KEY')
# or as a system environment variable.
GROQ_API_KEY = None
try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    print("Successfully loaded GROQ_API_KEY from Colab Secrets.")
except (ImportError, KeyError):
    print("Could not load key from Colab Secrets. Checking for environment variable.")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if GROQ_API_KEY:
        print("Successfully loaded GROQ_API_KEY from environment variable.")

# Using a top-tier, reliable model from Groq for best performance
MODEL_NAME = 'llama3-70b-8192'

# --- File Paths ---
DEV_DATA_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/dev_data.tsv'
DEV_LABELS_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/train_label.tsv'
PREDICTION_OUTPUT_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/output/prediction_subtask_1_llm_fewshot.tsv'

# --- Helper Functions ---
def robust_read_lines(file_path):
    """Reads a text file, trying different encodings."""
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = [line.strip() for line in f.readlines()]
            print(f"Successfully read {file_path} with encoding '{enc}'.")
            return lines
        except (UnicodeDecodeError, TypeError):
            continue
    print(f"FATAL: Could not read {file_path}.")
    return None

def load_and_prepare_data(data_path, labels_path):
    """Loads data and labels and merges them into a DataFrame."""
    try:
        questions = robust_read_lines(data_path)
        labels = robust_read_lines(labels_path)
        if questions is None or labels is None or len(questions) != len(labels):
            print(f"FATAL: Mismatch in line count or failed to read files.")
            return None

        merged_df = pd.DataFrame({'Question': questions, 'True_Labels': labels})
        print(f"\nData loaded successfully. Total questions to process: {len(merged_df)}")
        return merged_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure file paths are correct.")
        return None

def build_prompt(question):
    """
    Builds the advanced prompt with detailed instructions, category definitions,
    and few-shot examples.
    """
    # The user-provided descriptions are integrated directly into the prompt.
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
**Example 3:**
Question: "السلام عليكم مشكلتي تقتصر على تكرار كلمة معينة لمدة طويلة من الزمن مثلا اذا سمعت شخص قام بتكرار كلمة معينة ابقى اكررها مع نفسي احاول توقيف نفسي ولكن دون جدوى حتى اصاب بلاحباط اصبت بهذا الشي من قبل والحمد لله تخلصت منه ولكنه رجع مع العلم ان الوسواس شائع في عائلتنا قالت امي استغفري بدل تكرار وشكرا"
Reasoning: The user describes a specific, repetitive symptom (diagnosable as potential OCD/anxiety) and mentions self-help advice from their mother. This fits 'Diagnosis' and 'Healthy Lifestyle'.
Final Answer: A,E

---
**Example 4:**
Question: "اكتئاب وفوبيا من المجتمع وانعزال وانطوائية وتعامل بسلبية"
Reasoning: The user lists several symptoms and conditions. They are implicitly asking for help with these conditions. This is a call for 'Treatment' and also relates to the course of their disease ('Epidemiology').
Final Answer: B,D

---
**Example 5:**
Question: "هل الإحساس بقرب الاجل و الخوف من الموت و الاحلام من اعراض الاكتئاب و القلق؟! و كيف يمكنني تخطي هذه المرحلة لان حياتي اصبحت جحيم"
Reasoning: The user explicitly asks if their feelings are symptoms of depression ('Diagnosis'), asks how to overcome them ('Treatment'), and questions the prognosis of their condition ('Epidemiology').
Final Answer: A,B,D
---

**STEP 3: Classify the New Question (Your Task)**
Now, apply the same logic to the following question. Provide a step-by-step thought process and conclude with the 'Final Answer' on a new line.

**Question:** "{question}"

**Reasoning:**
Final Answer:"""
    return prompt

def get_prediction_from_groq(client, question, all_labels):
    """Gets a multi-label prediction using the advanced prompt."""
    full_prompt = build_prompt(question)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model=MODEL_NAME,
            temperature=0.1,  # Low temperature for high-fidelity reasoning
            max_tokens=1024,
            top_p=1,
        )
        response = chat_completion.choices[0].message.content.strip()

        # Use regex to robustly find the 'Final Answer' line
        match = re.search(r"Final Answer:\s*([A-Z,]+)", response, re.IGNORECASE)

        if match:
            # Extract the letters, split by comma, strip whitespace, and filter valid labels
            predicted_labels_str = match.group(1)
            predicted_labels = [label.strip() for label in predicted_labels_str.split(',') if label.strip() in all_labels]
            # Ensure no empty strings are in the list
            predicted_labels = [label for label in predicted_labels if label]
        else:
            # Fallback if the regex fails
            predicted_labels = []

        if not predicted_labels:
            print(f"Warning: Model did not return a valid, formatted label. Response was: '{response}'")
            return "" # Return empty string if no valid labels found

        return ",".join(sorted(predicted_labels)) # Sort for consistency

    except Exception as e:
        print(f"An API error occurred: {e}")
        return ""

def evaluate_predictions(true_labels, pred_labels, all_labels):
    """Calculates and prints the Weighted F1 and Jaccard scores."""
    mlb = MultiLabelBinarizer(classes=all_labels)
    true_labels_split = [[label.strip() for label in str(s).split(',')] for s in true_labels]
    pred_labels_split = [[label.strip() for label in str(s).split(',')] for s in pred_labels]

    y_true = mlb.fit_transform(true_labels_split)
    y_pred = mlb.transform(pred_labels_split)

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Jaccard Score:     {jaccard:.4f}")
    print("--------------------------\n")

# --- Main Execution ---
def main():
    print("Starting LLM-based Question Categorization with Few-Shot Prompting...")
    if not GROQ_API_KEY:
        print("FATAL: GROQ_API_KEY not found. Please set it in Colab Secrets or as an environment variable.")
        return

    dev_df = load_and_prepare_data(DEV_DATA_PATH, DEV_LABELS_PATH)
    if dev_df is None or dev_df.empty:
        print("Halting execution due to data loading issues.")
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        return

    # Discover all possible labels from the 'True_Labels' column
    all_labels = sorted([
        'A', 'B', 'C', 'D', 'E', 'F', 'Z'
    ])
    print(f"Using predefined labels: {all_labels}")

    print(f"\nGenerating predictions for {len(dev_df)} questions using '{MODEL_NAME}'...")
    predictions = []
    for _, row in tqdm(dev_df.iterrows(), total=dev_df.shape[0], desc="Classifying Questions"):
        prediction = get_prediction_from_groq(client, row['Question'], all_labels)
        predictions.append(prediction)

    dev_df['Predicted_Labels'] = predictions
    print("Prediction generation complete.")

    output_dir = os.path.dirname(PREDICTION_OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(PREDICTION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(dev_df['Predicted_Labels']))
    print(f"Predictions saved to '{PREDICTION_OUTPUT_PATH}'")

    evaluate_predictions(
        dev_df['True_Labels'].tolist(),
        dev_df['Predicted_Labels'].tolist(),
        all_labels
    )
    print("Script finished successfully.")

if __name__ == "__main__":
    main()
