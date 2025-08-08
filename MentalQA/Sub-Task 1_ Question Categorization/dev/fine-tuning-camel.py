# -*- coding: utf-8 -*-
"""
IMPROVED Multi-Label Arabic Mental Health Classification Model
- FIX: Correctly handles whitespace in labels (e.g., "A, B") to prevent errors.
- Addresses the conservative prediction issue and implements better multi-label strategies.
"""
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Import Hugging Face Transformers components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, classification_report

# --- Configuration ---
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- File Paths ---
# WARNING: This setup trains and predicts on the same data without a validation set.
# This is not ideal for checking model performance but matches the script's original logic.
DATA_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/dev_data.tsv'
LABELS_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/train_label.tsv'
TRAINING_OUTPUT_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/output/improved_camelbert_checkpoints'
PREDICTION_OUTPUT_PATH = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/output/predictions_improved_camelbert.tsv'

# --- Custom Model with Focal Loss ---
class ImprovedMultiLabelModel(nn.Module):
    # (This class is unchanged)
    def __init__(self, model_name, num_labels, alpha=1.0, gamma=2.0):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification", ignore_mismatched_sizes=True
        )
        self.alpha, self.gamma, self.num_labels = alpha, gamma, num_labels
    def focal_loss(self, logits, labels):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1-pt)**self.gamma * BCE_loss).mean()
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        logits = self.bert.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# --- Helper Functions ---
def robust_read_lines(file_path):
    # (This function is unchanged)
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_and_prepare_data(data_path, labels_path):
    # (This function is unchanged)
    questions, labels = robust_read_lines(data_path), robust_read_lines(labels_path)
    if len(questions) != len(labels):
        raise ValueError(f"Mismatch in line count between data and labels.")
    return pd.DataFrame({'text': questions, 'labels_str': labels})

# =================================================================================
# --- START OF THE FIX ---
# This new helper function ensures labels are cleaned correctly every time.
# =================================================================================
def process_label_strings(label_series):
    """Takes a pandas Series of label strings and cleans them."""
    processed_labels = []
    for s in label_series:
        # Split by comma, then strip whitespace from each individual part
        labels = [label.strip() for label in s.split(',') if label.strip()]
        processed_labels.append(labels)
    return processed_labels
# =================================================================================
# --- END OF THE FIX ---
# =================================================================================

def analyze_label_cooccurrence(labels_matrix, label_names):
    # (This function is unchanged)
    cooccurrence = np.dot(labels_matrix.T, labels_matrix)
    label_frequencies = np.sum(labels_matrix, axis=0)
    cooccurrence_prob = {}
    for i, label1 in enumerate(label_names):
        for j, label2 in enumerate(label_names):
            if i != j and label_frequencies[i] > 0:
                prob = cooccurrence[i, j] / label_frequencies[i]
                if prob > 0.3:
                    cooccurrence_prob[(label1, label2)] = prob
    return cooccurrence_prob

class ImprovedMentalQADataset(Dataset):
    # (This class is unchanged)
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

def adaptive_threshold_prediction(logits, label_names, cooccurrence_prob, base_threshold=0.3):
    # (This function is unchanged)
    probs = 1 / (1 + np.exp(-logits))
    predictions = []
    for i in range(len(probs)):
        sample_probs = probs[i]
        predicted_labels = {label_names[idx] for idx in np.where(sample_probs >= base_threshold)[0]}
        for label in list(predicted_labels):
            for idx, other_label in enumerate(label_names):
                if other_label not in predicted_labels and (label, other_label) in cooccurrence_prob:
                    cooccur_prob = cooccurrence_prob[(label, other_label)]
                    adjusted_threshold = base_threshold * (1 - cooccur_prob * 0.5)
                    if sample_probs[idx] >= adjusted_threshold:
                        predicted_labels.add(other_label)
        if not predicted_labels:
            predicted_labels.add(label_names[np.argmax(sample_probs)])
        if len(predicted_labels) > 4:
            label_prob_pairs = sorted([(label, sample_probs[label_names.index(label)]) for label in predicted_labels], key=lambda x: x[1], reverse=True)
            predicted_labels = {pair[0] for pair in label_prob_pairs[:4]}
        predictions.append(sorted(list(predicted_labels)))
    return predictions

# --- Main Execution ---
def main():
    print(f"Starting IMPROVED Multi-Label Classification with '{MODEL_NAME}'...")

    # 1. Load Data
    print("\n--- Loading Data ---")
    full_df = load_and_prepare_data(DATA_PATH, LABELS_PATH)
    if full_df is None: return
    train_df = full_df.copy()
    print(f"Using all {len(train_df)} examples for training.")

    # 2. Preprocess Labels
    print("\n--- Preprocessing Labels ---")
    # **FIX APPLIED HERE:** Use the helper function to discover all unique labels correctly.
    all_labels_flat = [label for sublist in process_label_strings(full_df['labels_str']) for label in sublist]
    all_labels = sorted(list(set(all_labels_flat)))
    print(f"Discovered {len(all_labels)} unique labels: {all_labels}")

    mlb = MultiLabelBinarizer(classes=all_labels)
    # **FIX APPLIED HERE:** Use the helper function to create the training labels correctly.
    # This will prevent the "unknown class" warning.
    train_labels = mlb.fit_transform(process_label_strings(train_df['labels_str']))
    print("Label processing complete.")

    # Analyze label co-occurrence patterns
    print("\n--- Analyzing Label Co-occurrence ---")
    cooccurrence_prob = analyze_label_cooccurrence(train_labels, all_labels)
    print(f"Found {len(cooccurrence_prob)} strong label co-occurrence patterns")

    # 3. Tokenize Text
    print("\n--- Tokenizing Text ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    train_dataset = ImprovedMentalQADataset(train_encodings, train_labels)

    # 4. Initialize Improved Model
    print("\n--- Initializing Improved Model ---")
    model = ImprovedMultiLabelModel(MODEL_NAME, len(all_labels), alpha=1.0, gamma=2.0).to(DEVICE)
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR, num_train_epochs=15, per_device_train_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=2e-5, warmup_steps=100,
        weight_decay=0.01, logging_dir='./logs', logging_steps=20, save_strategy="epoch",
        save_total_limit=3, dataloader_num_workers=2, fp16=True if torch.cuda.is_available() else False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    # 5. Fine-Tune the Model
    print("\n--- Starting Enhanced Fine-Tuning ---")
    trainer.train()
    print("Fine-tuning complete.")

    # 6. Generate Improved Predictions on All Data
    print("\n--- Generating Predictions for All Samples ---")
    predictions = trainer.predict(train_dataset)
    logits = predictions.predictions
    predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=0.25)
    pred_labels_str = [','.join(labels) for labels in predicted_labels_list]
    print(f"Generated {len(pred_labels_str)} predictions for submission")

    # Save predictions
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT_PATH), exist_ok=True)
    with open(PREDICTION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for label in pred_labels_str:
            f.write(f"{label}\n")
    print(f"Predictions saved to '{PREDICTION_OUTPUT_PATH}'")

    # 7. Detailed Evaluation on Training Data
    print("\n--- Training Data Performance Analysis ---")
    # **FIX APPLIED HERE:** Use the helper function to get the true binary labels for comparison.
    predicted_labels_binary = mlb.transform(predicted_labels_list)
    f1_weighted = f1_score(train_labels, predicted_labels_binary, average='weighted', zero_division=0)
    print(f"Weighted F1 Score on training data: {f1_weighted:.4f}")
    print("\n--- Per-Label Performance ---")
    print(classification_report(train_labels, predicted_labels_binary, target_names=all_labels, zero_division=0))

if __name__ == "__main__":
    main()
