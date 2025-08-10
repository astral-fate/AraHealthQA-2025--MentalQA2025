# Cell 1: Installations
# !pip install transformers[torch] accelerate -U
# !pip install optuna

# Cell 2: Imports
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import shutil
import optuna

# Import from Google Colab's drive module
from google.colab import drive

# Import Hugging Face Transformers components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_callback import EarlyStoppingCallback

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# Cell 3: Mount Drive and Define Paths
drive.mount('/content/drive')

# --- Configuration ---
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- File Paths for Google Drive ---
BASE_DRIVE_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/'
DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'dev_data.tsv')
LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'train_label.tsv')
TUNING_OUTPUT_DIR = os.path.join(BASE_DRIVE_DIR, 'tuning_output')
FINAL_MODEL_DIR = os.path.join(BASE_DRIVE_DIR, 'final_model')

# Create directories in your Google Drive if they don't exist
os.makedirs(TUNING_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Cell 4: Custom Model Definition
class ImprovedMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels, alpha=1.0, gamma=2.0):
        super().__init__()
        # Load the pre-trained model, ignoring size mismatches in the classifier layer
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        self.alpha, self.gamma, self.num_labels = alpha, gamma, num_labels

    def focal_loss(self, logits, labels):
        # A numerically stable implementation of Focal Loss
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
            loss = self.focal_loss(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



# Cell 5: Helper Functions
def robust_read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_and_prepare_data(data_path, labels_path):
    questions = robust_read_lines(data_path)
    labels = robust_read_lines(labels_path)
    if len(questions) != len(labels):
        raise ValueError("Mismatch in line count between data and labels.")
    return pd.DataFrame({'text': questions, 'labels_str': labels})

def process_label_strings(label_series):
    return [
        [label.strip() for label in s.split(',') if label.strip()]
        for s in label_series
    ]

def analyze_label_cooccurrence(labels_matrix, label_names):
    cooccurrence_matrix = np.dot(labels_matrix.T, labels_matrix)
    label_frequencies = np.sum(labels_matrix, axis=0)
    cooccurrence_prob = {}
    for i, label1 in enumerate(label_names):
        for j, label2 in enumerate(label_names):
            if i != j and label_frequencies[i] > 0:
                # Calculate P(label2 | label1)
                prob = cooccurrence_matrix[i, j] / label_frequencies[i]
                if prob > 0.3: # Only consider strong correlations
                    cooccurrence_prob[(label1, label2)] = prob
    return cooccurrence_prob

class MentalQADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def adaptive_threshold_prediction(logits, label_names, cooccurrence_prob, base_threshold=0.5):
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


# Cell 6: Main Execution Logic
# Global variables to be set by the initial data load
mlb = None
all_labels = None
cooccurrence_prob = None
train_dataset = None
dev_dataset = None

def objective(trial: optuna.Trial):
    """
    This function defines one trial of the hyperparameter search.
    Optuna will call this function multiple times with different parameter combinations.
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 7e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 5, 20)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    focal_alpha = trial.suggest_float("focal_alpha", 0.25, 1.5)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)

    # Use a temporary directory inside the main tuning output folder
    trial_output_dir = os.path.join(TUNING_OUTPUT_DIR, f"trial_{trial.number}")

    model = ImprovedMultiLabelModel(
        MODEL_NAME,
        num_labels=len(all_labels),
        alpha=focal_alpha,
        gamma=focal_gamma
    ).to(DEVICE)

    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=True, # Always True on Colab GPU
        save_total_limit=1,
    )

    def compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        base_threshold = trial.suggest_float("base_threshold", 0.2, 0.6)
        predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=base_threshold)
        y_pred = mlb.transform(predicted_labels_list)
        y_true = labels.astype(int)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {'f1_weighted': f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=dev_dataset)

    shutil.rmtree(trial_output_dir)
    return eval_metrics['eval_f1_weighted']


def main():
    global mlb, all_labels, cooccurrence_prob, train_dataset, dev_dataset

    print(f"Starting Multi-Label Classification for '{MODEL_NAME}'")
    print("This script will first find the best hyperparameters and then train a final model.")

    print("\n--- 1. Loading and Splitting Data ---")
    try:
        full_df = load_and_prepare_data(DATA_PATH, LABELS_PATH)
    except FileNotFoundError:
        print(f"\nERROR: Data files not found in your Google Drive at {BASE_DRIVE_DIR}")
        return

    train_df, dev_df = train_test_split(full_df, test_size=50, random_state=42, shuffle=True)
    print(f"Using {len(train_df)} samples for training and {len(dev_df)} for development.")

    print("\n--- 2. Preprocessing Labels ---")
    all_labels_flat = [label for sublist in process_label_strings(full_df['labels_str']) for label in sublist]
    all_labels = sorted(list(set(all_labels_flat)))
    print(f"Discovered {len(all_labels)} unique labels: {all_labels}")

    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit(process_label_strings(full_df['labels_str']))
    train_labels = mlb.transform(process_label_strings(train_df['labels_str']))
    dev_labels = mlb.transform(process_label_strings(dev_df['labels_str']))
    cooccurrence_prob = analyze_label_cooccurrence(train_labels, all_labels)
    print(f"Found {len(cooccurrence_prob)} strong label co-occurrence patterns.")

    print("\n--- 3. Tokenizing Text ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    dev_encodings = tokenizer(dev_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    train_dataset = MentalQADataset(train_encodings, train_labels)
    dev_dataset = MentalQADataset(dev_encodings, dev_labels)

    print("\n--- 4. Starting Hyperparameter Optimization with Optuna ---")
    study = optuna.create_study(direction="maximize", study_name="MentalQA_Optimization")
    study.optimize(objective, n_trials=30)

    print("\nOptimization Finished!")
    print(f"Best trial F1 Score: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  - {key}: {value}")

    print("\n--- 5. Training Final Model with Best Hyperparameters ---")
    final_model = ImprovedMultiLabelModel(
        MODEL_NAME,
        num_labels=len(all_labels),
        alpha=best_params['focal_alpha'],
        gamma=best_params['focal_gamma']
    ).to(DEVICE)

    final_training_args = TrainingArguments(
        output_dir=FINAL_MODEL_DIR,
        num_train_epochs=best_params['num_train_epochs'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=True,
        save_total_limit=1,
    )

    def final_compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=best_params['base_threshold'])
        y_pred = mlb.transform(predicted_labels_list)
        y_true = labels.astype(int)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {'f1_weighted': f1}

    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=final_compute_metrics,
    )

    final_trainer.train()

    print("\n--- 6. Final Performance Analysis on Development Set ---")
    predictions = final_trainer.predict(dev_dataset)
    final_logits = predictions.predictions
    final_predicted_labels_list = adaptive_threshold_prediction(final_logits, all_labels, cooccurrence_prob, base_threshold=best_params['base_threshold'])
    final_predicted_labels_binary = mlb.transform(final_predicted_labels_list)

    final_f1_weighted = f1_score(dev_labels, final_predicted_labels_binary, average='weighted', zero_division=0)
    print(f"\nFinal Weighted F1 Score on dev data: {final_f1_weighted:.4f}")

    print("\n--- Final Per-Label Performance on Dev Set ---")
    print(classification_report(dev_labels, final_predicted_labels_binary, target_names=all_labels, zero_division=0))
    print(f"\nBest model saved in your Google Drive at: {FINAL_MODEL_DIR}")

# Run the main function
if __name__ == '__main__':
    main()
