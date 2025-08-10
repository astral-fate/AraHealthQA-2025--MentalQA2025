# -*- coding: utf-8 -*-
"""
evaluate_arabert_on_test_set.ipynb

This script loads a fine-tuned AraBERT model from a specific checkpoint
and evaluates its performance on the designated test set.
"""

# Cell 1: Installations
# Ensure necessary libraries are installed in the environment.
# !pip install transformers[torch] accelerate scikit-learn pandas safetensors

# Cell 2: Imports
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
from transformers.modeling_outputs import SequenceClassifierOutput

# Import scikit-learn utilities
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# Cell 3: Mount Drive and Define All Paths
print("🗂️ Mounting Google Drive...")
drive.mount('/content/drive')

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Base Paths ---
BASE_DRIVE_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/'

# --- Model Configuration ---
# The original Hugging Face model name used for training
BASE_MODEL_NAME = "aubmindlab/bert-base-arabertv2"
# The specific checkpoint from your training run that you want to evaluate
CHECKPOINT_TO_LOAD = "checkpoint-380"
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DRIVE_DIR, 'final_model', CHECKPOINT_TO_LOAD)
WEIGHTS_PATH = os.path.join(MODEL_CHECKPOINT_PATH, 'model.safetensors') # Using safetensors for secure loading

# --- Data Paths ---
# Use the ORIGINAL training data to fit the binarizer and calculate co-occurrence
TRAIN_DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'dev_data.tsv')
TRAIN_LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'train_label.tsv')

# The TEST data for final evaluation (150 samples)
TEST_DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'data/subtask1_input_test.tsv')
TEST_LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'data/subtask1_output_test.tsv')

# --- Output Path ---
# Directory to save the final prediction results
RESULTS_DIR = os.path.join(BASE_DRIVE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# Cell 4: Custom Model and Helper Functions (from your training script)
class ImprovedMultiLabelModel(nn.Module):
    """
    The exact same custom model class used during training, including Focal Loss.
    This ensures the architecture matches the saved weights.
    """
    def __init__(self, model_name, num_labels, alpha=1.0, gamma=2.0):
        super().__init__()
        # Load the base model structure
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        # Store focal loss parameters, though they are not used in evaluation
        self.alpha, self.gamma, self.num_labels = alpha, gamma, num_labels

    def focal_loss(self, logits, labels):
        # This function is not called during prediction but is part of the model definition
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1-pt)**self.gamma * BCE_loss).mean()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the raw BERT outputs
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Use the [CLS] token's representation for classification
        pooled_output = sequence_output[:, 0]
        logits = self.bert.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Loss calculation is skipped during inference but shown here for completeness
            loss = self.focal_loss(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

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
                if prob > 0.3: # Only consider strong correlations
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


# Cell 5: Main Evaluation Function
def evaluate_on_test_set():
    """
    Main function to load the fine-tuned AraBERT model and evaluate it on the test set.
    """
    print("🚀 Starting Evaluation of Fine-Tuned AraBERT Model on the Test Set...")

    print(f"\n--- 1. Loading Base Tokenizer and Model from '{BASE_MODEL_NAME}' ---")
    try:
        # Step 1: Load the tokenizer from the original source.
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print("✅ Tokenizer loaded successfully.")

        # Step 2: Load the training data to determine the number of labels.
        full_train_df = load_and_prepare_data(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
        all_labels_flat = [label for sublist in process_label_strings(full_train_df['labels_str']) for label in sublist]
        all_labels = sorted(list(set(all_labels_flat)))
        NUM_LABELS = len(all_labels)
        print(f"Discovered {NUM_LABELS} unique labels from training data.")

        # Step 3: Instantiate the model architecture.
        print("Instantiating model architecture...")
        model = ImprovedMultiLabelModel(model_name=BASE_MODEL_NAME, num_labels=NUM_LABELS)

        # Step 4: Load the fine-tuned weights from your specific checkpoint.
        print(f"Loading fine-tuned weights from: {WEIGHTS_PATH}")
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights file not found at {WEIGHTS_PATH}. Please ensure the checkpoint path is correct.")
        state_dict = load_file(WEIGHTS_PATH, device=DEVICE.type)

        # Step 5: Apply the loaded weights to the model structure.
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        print("✅ Model architecture created and fine-tuned weights applied successfully.")

    except Exception as e:
        print(f"❌ FATAL ERROR during model loading: {e}")
        return

    print("\n--- 2. Preprocessing Labels and Co-occurrence from Training Data ---")
    mlb = MultiLabelBinarizer(classes=all_labels).fit(process_label_strings(full_train_df['labels_str']))
    
    # We use a train/dev split of the original data to calculate co-occurrence, just like in training.
    train_df, _ = train_test_split(full_train_df, test_size=50, random_state=42, shuffle=True)
    train_labels_binary = mlb.transform(process_label_strings(train_df['labels_str']))
    cooccurrence_prob = analyze_label_cooccurrence(train_labels_binary, all_labels)
    print(f"Calculated {len(cooccurrence_prob)} strong label co-occurrence patterns.")

    print("\n--- 3. Loading and Tokenizing Test Data ---")
    test_df = load_and_prepare_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    print(f"Loaded {len(test_df)} samples from the test set.")
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    test_dataset = MentalQADataset(test_encodings)

    print("\n--- 4. Generating Predictions for the Test Set ---")
    # A basic Trainer is used here just as a predictor. No training arguments are needed.
    trainer = Trainer(model=model)
    raw_predictions = trainer.predict(test_dataset)
    logits = raw_predictions.predictions

    print("\n--- 5. Post-processing Predictions with Adaptive Thresholding ---")
    # ⚠️ IMPORTANT: Replace this with the 'base_threshold' from your Optuna study results!
    best_base_threshold = 0.45 # <--- REPLACE THIS VALUE
    print(f"Using the best base_threshold found during tuning: {best_base_threshold:.4f}")
    predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=best_base_threshold)

    print("\n--- 6. Final Evaluation on the Test Set ---")
    y_true_binary = mlb.transform(process_label_strings(test_df['labels_str']))
    y_pred_binary = mlb.transform(predicted_labels_list)
    weighted_f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    
    print("\n--- 🥁 Final Test Set Results 🥁 ---")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("------------------------------------\n")
    print("--- Per-Label Performance (Test Set) ---")
    print(classification_report(y_true_binary, y_pred_binary, target_names=all_labels, zero_division=0))

    print("\n--- 7. Saving Predictions to File ---")
    test_df['Predicted_Labels'] = [",".join(p) for p in predicted_labels_list]
    prediction_output_path = os.path.join(RESULTS_DIR, f"arabert_{CHECKPOINT_TO_LOAD}_test_predictions.tsv")
    test_df[['Predicted_Labels']].to_csv(prediction_output_path, sep='\t', header=False, index=False)
    print(f"💾 Test set predictions saved to: {prediction_output_path}")

    print("\n✅ Evaluation complete.")


# Cell 6: Run the Evaluation
if __name__ == '__main__':
    evaluate_on_test_set()
