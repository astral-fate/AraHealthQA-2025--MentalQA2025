
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from google.colab import drive
from safetensors.torch import load_file

# Import Hugging Face Transformers components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, classification_report
from sklearn.model_selection import train_test_split

# Cell 2: Mount Drive and Define Paths
drive.mount('/content/drive')

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- File Paths ---
BASE_DRIVE_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/'
# This is the original model name from Hugging Face
BASE_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

# Path to the specific checkpoint containing the weights
MODEL_CHECKPOINT_TO_LOAD = os.path.join(BASE_DRIVE_DIR, 'final_model/checkpoint-152')
WEIGHTS_PATH = os.path.join(MODEL_CHECKPOINT_TO_LOAD, 'model.safetensors')


# Paths for TEST data
TEST_DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'data/subtask1_input_test.tsv')
TEST_LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'data/subtask1_output_test.tsv')

# Paths for original TRAINING data
TRAIN_DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'dev_data.tsv')
TRAIN_LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'train_label.tsv')

# Directory to SAVE the final prediction results
RESULTS_DIR = os.path.join(BASE_DRIVE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# Cell 3: Helper Functions & Model Class
class ImprovedMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels, alpha=1.0, gamma=2.0):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        self.alpha, self.gamma = alpha, gamma
    def focal_loss(self, logits, labels):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt)**self.gamma * bce_loss).mean()
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.bert.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

def robust_read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_and_prepare_data(data_path, labels_path=None):
    questions = robust_read_lines(data_path)
    if labels_path:
        labels = robust_read_lines(labels_path)
        if len(questions) != len(labels):
            raise ValueError("Mismatch in line count between data and labels.")
        return pd.DataFrame({'text': questions, 'labels_str': labels})
    return pd.DataFrame({'text': questions})

def process_label_strings(label_series):
    return [[label.strip() for label in s.split(',') if label.strip()] for s in label_series]

def analyze_label_cooccurrence(labels_matrix, label_names):
    cooccurrence_matrix = np.dot(labels_matrix.T, labels_matrix)
    label_frequencies = np.sum(labels_matrix, axis=0)
    cooccurrence_prob = {}
    for i, label1 in enumerate(label_names):
        for j, label2 in enumerate(label_names):
            if i != j and label_frequencies[i] > 0:
                prob = cooccurrence_matrix[i, j] / label_frequencies[i]
                if prob > 0.3:
                    cooccurrence_prob[(label1, label2)] = prob
    return cooccurrence_prob

class MentalQADataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

def adaptive_threshold_prediction(logits, label_names, cooccurrence_prob, base_threshold=0.5):
    probs = 1 / (1 + np.exp(-logits))
    predictions = []
    for i in range(len(probs)):
        sample_probs = probs[i]
        predicted_labels = {label_names[idx] for idx in np.where(sample_probs >= base_threshold)[0]}
        for label in list(predicted_labels):
            for idx, other_label in enumerate(label_names):
                if other_label not in predicted_labels and (label, other_label) in cooccurrence_prob:
                    cooccur_prob = cooccurrence_prob.get((label, other_label), 0)
                    adjusted_threshold = base_threshold * (1 - cooccur_prob * 0.5)
                    if sample_probs[idx] >= adjusted_threshold:
                        predicted_labels.add(other_label)
        if not predicted_labels:
            predicted_labels.add(label_names[np.argmax(sample_probs)])
        if len(predicted_labels) > 4:
            label_prob_pairs = sorted([(l, sample_probs[label_names.index(l)]) for l in predicted_labels], key=lambda x: x[1], reverse=True)
            predicted_labels = {p[0] for p in label_prob_pairs[:4]}
        predictions.append(sorted(list(predicted_labels)))
    return predictions

# Cell 4: Final Evaluation Script
def test_model():
    """Main function to load the fine-tuned model and evaluate it on the test set."""
    print("🚀 Starting Evaluation of Fine-Tuned Model on the Test Set...")
    NUM_LABELS = 7

    # =========================================================================================
    # === THE DEFINITIVE FIX: Load Tokenizer and Base Model from Hub, then apply local weights ===
    # =========================================================================================
    print(f"\n--- 1. Loading tokenizer and model from base '{BASE_MODEL_NAME}' ---")
    try:
        # Step 1: Load the tokenizer from the original source. This is safe.
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print("✅ Tokenizer loaded successfully.")

        # Step 2: Create the model's architecture from the original source.
        print("Instantiating model architecture...")
        model = ImprovedMultiLabelModel(model_name=BASE_MODEL_NAME, num_labels=NUM_LABELS)

        # Step 3: Load the fine-tuned weights from your local safetensors file.
        print(f"Loading fine-tuned weights from: {WEIGHTS_PATH}")
        state_dict = load_file(WEIGHTS_PATH, device=DEVICE.type)

        # Step 4: Apply the loaded weights to the model structure.
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        print("✅ Model architecture created and fine-tuned weights applied successfully.")

    except Exception as e:
        print(f"❌ Error during the manual loading process: {e}")
        return

    # 2. Load and Prepare Data
    print("\n--- 2. Loading datasets ---")
    test_df = load_and_prepare_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    full_train_df = load_and_prepare_data(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
    train_df, _ = train_test_split(full_train_df, test_size=50, random_state=42, shuffle=True)

    # 3. Preprocess Labels and Co-occurrence
    print("\n--- 3. Preprocessing labels for evaluation ---")
    all_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'Z']
    mlb = MultiLabelBinarizer(classes=all_labels).fit(process_label_strings(full_train_df['labels_str']))
    train_labels_binary = mlb.transform(process_label_strings(train_df['labels_str']))
    cooccurrence_prob = analyze_label_cooccurrence(train_labels_binary, all_labels)

    # 4. Tokenize Test Data and Predict
    print("\n--- 4. Generating predictions for the test set ---")
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    test_dataset = MentalQADataset(test_encodings)
    trainer = Trainer(model=model)
    raw_predictions = trainer.predict(test_dataset)
    logits = raw_predictions.predictions

    # 5. Post-process Predictions
    print("\n--- 5. Applying adaptive thresholding to predictions ---")
    best_threshold = 0.2462205131750359
    print(f"Using the best base_threshold found during tuning: {best_threshold:.4f}")
    predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=best_threshold)

    # 6. Evaluate Final Predictions
    print("\n--- 6. Final Evaluation on the Test Set ---")
    y_true_binary = mlb.transform(process_label_strings(test_df['labels_str']))
    y_pred_binary = mlb.transform(predicted_labels_list)
    weighted_f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)

    print("\n--- 🥁 Final Test Set Results 🥁 ---")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Jaccard Score:     {jaccard:.4f}")
    print("------------------------------------\n")
    print("--- Per-Label Performance (Test Set) ---")
    print(classification_report(y_true_binary, y_pred_binary, target_names=all_labels, zero_division=0))

    # 7. Save Predictions
    test_df['Predicted_Labels'] = [",".join(p) for p in predicted_labels_list]
    prediction_output_path = os.path.join(RESULTS_DIR, "camelbert_finetuned_test_predictions.tsv")
    test_df[['Predicted_Labels']].to_csv(prediction_output_path, sep='\t', header=False, index=False)
    print(f"💾 Test set predictions saved to: {prediction_output_path}")
    print("\n✅ Evaluation complete.")

# Run the testing function
test_model()





