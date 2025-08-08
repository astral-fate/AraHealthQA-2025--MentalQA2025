import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import shutil

# Import Hugging Face Transformers components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold

# --- Configuration ---
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
BASE_DATA_DIR = 'D:/mental/data'
BASE_OUTPUT_DIR = 'D:/mental/output'

# Create the main output directory if it doesn't exist
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Construct full paths using os.path.join for cross-platform compatibility
DATA_PATH = os.path.join(BASE_DATA_DIR, 'D:/mental/data/dev_data.tsv')
LABELS_PATH = os.path.join(BASE_DATA_DIR, 'D:/mental/data/train_label.tsv')
TRAINING_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'D:/mental/output/camel_99_with_validation')

# Create the main output directory if it doesn't exist
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_DATA_DIR, exist_ok=True)

# NOTE: You will need to place your 'dev_data.tsv' and 'train_label.tsv' into a 'data' subfolder.
# For demonstration, I am creating dummy files.
with open(os.path.join(BASE_DATA_DIR, 'dev_data.tsv'), 'w', encoding='utf-8') as f:
    for i in range(350):
        f.write(f"هذا نص سؤال تجريبي رقم {i}\n")
with open(os.path.join(BASE_DATA_DIR, 'train_label.tsv'), 'w', encoding='utf-8') as f:
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'Z']
    for i in range(350):
        num_labels = np.random.randint(1, 4)
        sample_labels = np.random.choice(labels, num_labels, replace=False)
        f.write(','.join(sample_labels) + '\n')


# Construct full paths using os.path.join for cross-platform compatibility
DATA_PATH = os.path.join(BASE_DATA_DIR, 'dev_data.tsv')
LABELS_PATH = os.path.join(BASE_DATA_DIR, 'train_label.tsv')
TRAINING_OUTPUT_DIR_BASE = os.path.join(BASE_OUTPUT_DIR, 'camel_kfold_validation')

# --- Custom Model with Focal Loss (Unchanged) ---
class ImprovedMultiLabelModel(nn.Module):
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

# --- Helper Functions (Unchanged) ---
def robust_read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_and_prepare_data(data_path, labels_path):
    questions, labels = robust_read_lines(data_path), robust_read_lines(labels_path)
    if len(questions) != len(labels):
        raise ValueError(f"Mismatch in line count between data and labels.")
    return pd.DataFrame({'text': questions, 'labels_str': labels})

def process_label_strings(label_series):
    processed_labels = []
    for s in label_series:
        labels = [label.strip() for label in s.split(',') if label.strip()]
        processed_labels.append(labels)
    return processed_labels

def analyze_label_cooccurrence(labels_matrix, label_names):
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
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

def adaptive_threshold_prediction(logits, label_names, cooccurrence_prob, base_threshold=0.3):
    probs = 1 / (1 + np.exp(-logits))
    predictions = []
    for i in range(len(probs)):
        sample_probs = probs[i]
        predicted_labels = {label_names[idx] for idx in np.where(sample_probs >= base_threshold)[0]}
        # Adjust based on co-occurrence
        for label in list(predicted_labels):
            for idx, other_label in enumerate(label_names):
                if other_label not in predicted_labels and (label, other_label) in cooccurrence_prob:
                    cooccur_prob = cooccurrence_prob[(label, other_label)]
                    adjusted_threshold = base_threshold * (1 - cooccur_prob * 0.5)
                    if sample_probs[idx] >= adjusted_threshold:
                        predicted_labels.add(other_label)
        # Ensure at least one prediction
        if not predicted_labels:
            predicted_labels.add(label_names[np.argmax(sample_probs)])
        # Limit max predictions
        if len(predicted_labels) > 4:
            label_prob_pairs = sorted([(label, sample_probs[label_names.index(label)]) for label in predicted_labels], key=lambda x: x[1], reverse=True)
            predicted_labels = {pair[0] for pair in label_prob_pairs[:4]}
        predictions.append(sorted(list(predicted_labels)))
    return predictions

# --- Main Execution with K-Fold Cross-Validation ---
def main():
    print(f"Starting Multi-Label Classification with K-Fold Cross-Validation for '{MODEL_NAME}'...")

    # 1. Load Data
    print("\n--- Loading Data ---")
    full_df = load_and_prepare_data(DATA_PATH, LABELS_PATH)
    full_df = full_df.reset_index(drop=True) # Ensure default indexing

    # 2. Preprocess All Labels Once
    print("\n--- Preprocessing Labels ---")
    all_labels_flat = [label for sublist in process_label_strings(full_df['labels_str']) for label in sublist]
    all_labels = sorted(list(set(all_labels_flat)))
    print(f"Discovered {len(all_labels)} unique labels: {all_labels}")
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit(process_label_strings(full_df['labels_str']))

    # 3. K-Fold Cross-Validation Setup
    N_SPLITS = 5
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Store out-of-fold predictions and true labels for final evaluation
    oof_preds = []
    oof_true = []
    oof_indices = []


    # 4. Iterate Through Folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df)):
        print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")
        
        # Clean up output directory for the fold
        fold_output_dir = f"{TRAINING_OUTPUT_DIR_BASE}_fold_{fold+1}"
        if os.path.exists(fold_output_dir):
            shutil.rmtree(fold_output_dir)

        # Split data for the current fold
        train_df = full_df.iloc[train_idx]
        val_df = full_df.iloc[val_idx]
        print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")

        # Transform labels for the current fold
        train_labels = mlb.transform(process_label_strings(train_df['labels_str']))
        val_labels = mlb.transform(process_label_strings(val_df['labels_str']))

        # Analyze co-occurrence ONLY on the fold's training data
        cooccurrence_prob = analyze_label_cooccurrence(train_labels, all_labels)
        print(f"Found {len(cooccurrence_prob)} strong label co-occurrence patterns for this fold.")

        # Tokenize text for the current fold
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=256)
        val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=256)

        train_dataset = ImprovedMentalQADataset(train_encodings, train_labels)
        val_dataset = ImprovedMentalQADataset(val_encodings, val_labels)

        # Define metrics computation function for this fold
        def compute_metrics(p):
            logits, labels = p.predictions, p.label_ids
            predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=0.25)
            y_pred = mlb.transform(predicted_labels_list)
            y_true = labels.astype(int)
            return {'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)}

        # Initialize a new model for each fold
        print("\n--- Initializing New Model for Fold ---")
        model = ImprovedMultiLabelModel(MODEL_NAME, len(all_labels), alpha=1.0, gamma=2.0).to(DEVICE)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=15, # Reduced epochs per fold; total training epochs will be 8*5=40
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            warmup_steps=50,
            weight_decay=0.05,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=1,
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Fine-tune the model on the fold's data
        print(f"\n--- Starting Fine-Tuning for Fold {fold+1} ---")
        trainer.train()

        # Generate predictions on the validation set for this fold
        print("\n--- Generating Predictions on Validation Set for Fold ---")
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions
        
        # Store predictions and true labels
        predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=0.25)
        oof_preds.extend(predicted_labels_list)
        oof_true.extend(val_df['labels_str'].tolist())
        oof_indices.extend(val_idx)

    # 5. Final Evaluation
    print("\n\n===== Overall K-Fold Performance Analysis =====")

    # Reorder predictions to match original dataframe order
    oof_preds_array = np.array(oof_preds, dtype=object)
    oof_true_array = np.array(oof_true, dtype=object)
    oof_indices_array = np.array(oof_indices)

    order = np.argsort(oof_indices_array)
    ordered_preds = oof_preds_array[order]
    ordered_true_str = oof_true_array[order]

    # Binarize both true and predicted labels for scoring
    y_true_final = mlb.transform(process_label_strings(pd.Series(ordered_true_str)))
    y_pred_final = mlb.transform(ordered_preds)

    f1_weighted_overall = f1_score(y_true_final, y_pred_final, average='weighted', zero_division=0)
    print(f"\nOverall Weighted F1 Score across all folds: {f1_weighted_overall:.4f}")

    print("\n--- Overall Per-Label Performance (based on out-of-fold predictions) ---")
    print(classification_report(y_true_final, y_pred_final, target_names=all_labels, zero_division=0))


if __name__ == "__main__":
    main()
