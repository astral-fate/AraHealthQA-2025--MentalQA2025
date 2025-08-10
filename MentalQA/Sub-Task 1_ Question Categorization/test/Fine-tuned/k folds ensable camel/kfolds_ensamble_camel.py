
# Mount Google Drive to access your files
from google.colab import drive
drive.mount('/content/drive')


# =================================================================================
# Cell 2: Main Script
# =================================================================================
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import shutil
from google.colab import drive

# Import Hugging Face Transformers components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold

# --- Configuration ---
# MODIFIED: Updated model name based on your logs
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- File Paths for Google Drive ---
# MODIFIED: Paths updated for Google Colab environment
BASE_DRIVE_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/'
DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'dev_data.tsv')
LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'train_label.tsv')
TRAINING_OUTPUT_DIR_BASE = os.path.join(BASE_DRIVE_DIR, 'output/camel_kfold_validation')

# Create the main output directory if it doesn't exist
os.makedirs(os.path.dirname(TRAINING_OUTPUT_DIR_BASE), exist_ok=True)


# --- Custom Model with Focal Loss (Unchanged) ---
class ImprovedMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels, alpha=1.0, gamma=2.0):
        super().__init__()
        # NOTE: The warning you saw about mismatched sizes is expected when you adapt a model
        # to a new task with a different number of labels.
        # `ignore_mismatched_sizes=True` correctly handles this by re-initializing the final classification layer.
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
    print("\n--- Loading Data from Google Drive---")
    full_df = load_and_prepare_data(DATA_PATH, LABELS_PATH)
    full_df = full_df.reset_index(drop=True)

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

    oof_preds, oof_true, oof_indices = [], [], []

    # 4. Iterate Through Folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df)):
        print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

        fold_output_dir = f"{TRAINING_OUTPUT_DIR_BASE}_fold_{fold+1}"
        if os.path.exists(fold_output_dir):
            shutil.rmtree(fold_output_dir)

        train_df, val_df = full_df.iloc[train_idx], full_df.iloc[val_idx]
        print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")

        train_labels = mlb.transform(process_label_strings(train_df['labels_str']))
        val_labels = mlb.transform(process_label_strings(val_df['labels_str']))

        cooccurrence_prob = analyze_label_cooccurrence(train_labels, all_labels)
        print(f"Found {len(cooccurrence_prob)} strong label co-occurrence patterns for this fold.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=256)
        val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=256)

        train_dataset = ImprovedMentalQADataset(train_encodings, train_labels)
        val_dataset = ImprovedMentalQADataset(val_encodings, val_labels)

        # MODIFIED: Define new base_threshold from your logs
        new_base_threshold = 0.3434835813289709

        def compute_metrics(p):
            logits, labels = p.predictions, p.label_ids
            predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=new_base_threshold)
            y_pred = mlb.transform(predicted_labels_list)
            y_true = labels.astype(int)
            return {'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)}

        print("\n--- Initializing New Model for Fold ---")
        # MODIFIED: Initialize model with new focal loss alpha and gamma from your logs
        model = ImprovedMultiLabelModel(
            MODEL_NAME,
            len(all_labels),
            alpha=1.194492474673312,
            gamma=2.8990426579607704
        ).to(DEVICE)

        # MODIFIED: Training arguments updated with new hyperparameters from your logs
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=10,
            learning_rate=3.26662135376377e-05,
            weight_decay=0.0199876722361212,
            per_device_train_batch_size=8,
            warmup_steps=50,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=1,
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)

        print(f"\n--- Starting Fine-Tuning for Fold {fold+1} ---")
        trainer.train()

        print("\n--- Generating Predictions on Validation Set for Fold ---")
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions

        # MODIFIED: Use the new base_threshold for out-of-fold predictions
        predicted_labels_list = adaptive_threshold_prediction(logits, all_labels, cooccurrence_prob, base_threshold=new_base_threshold)
        oof_preds.extend(predicted_labels_list)
        oof_true.extend(val_df['labels_str'].tolist())
        oof_indices.extend(val_idx)

    # 5. Final Evaluation
    print("\n\n===== Overall K-Fold Performance Analysis =====")
    oof_preds_array = np.array(oof_preds, dtype=object)
    oof_true_array = np.array(oof_true, dtype=object)
    oof_indices_array = np.array(oof_indices)

    order = np.argsort(oof_indices_array)
    ordered_preds = oof_preds_array[order]
    ordered_true_str = oof_true_array[order]

    y_true_final = mlb.transform(process_label_strings(pd.Series(ordered_true_str)))
    y_pred_final = mlb.transform(ordered_preds)

    f1_weighted_overall = f1_score(y_true_final, y_pred_final, average='weighted', zero_division=0)
    print(f"\nOverall Weighted F1 Score across all folds: {f1_weighted_overall:.4f}")

    print("\n--- Overall Per-Label Performance (based on out-of-fold predictions) ---")
    print(classification_report(y_true_final, y_pred_final, target_names=all_labels, zero_division=0))


if __name__ == "__main__":
    main()



# =================================================================================
# Cell 1: Imports and Setup (No changes needed)
# =================================================================================
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from google.colab import drive
from safetensors.torch import load_file
import glob # Used to find checkpoint directories

# Import Hugging Face Transformers components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, classification_report

# --- Mount Drive if not already mounted ---
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# =================================================================================
# Cell 2: Configuration and Paths
# =================================================================================

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
N_SPLITS = 5 # The number of folds you used for training

# --- File Paths ---
BASE_DRIVE_DIR = '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/'
BASE_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

# Path to the BASE directory where all K-Fold models were saved
K_FOLD_MODELS_DIR = os.path.join(BASE_DRIVE_DIR, 'output/camel_kfold_validation')

# Paths for TEST data (where we will evaluate)
TEST_DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'data/subtask1_input_test.tsv')
TEST_LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'data/subtask1_output_test.tsv')

# Paths for original TRAINING data (to build co-occurrence map)
TRAIN_DATA_PATH = os.path.join(BASE_DRIVE_DIR, 'dev_data.tsv')
TRAIN_LABELS_PATH = os.path.join(BASE_DRIVE_DIR, 'train_label.tsv')

# Directory to SAVE the final prediction results
RESULTS_DIR = os.path.join(BASE_DRIVE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =================================================================================
# Cell 3: Helper Functions & Model Class (Copied from your script)
# =================================================================================

class ImprovedMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels, alpha=1.0, gamma=2.0):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        # Note: We don't need focal loss for inference, but the class structure must match
        # how it was saved.
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # We only need the logits for prediction
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

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

def find_best_checkpoint(fold_dir):
    """Finds the path to the best checkpoint in a fold directory."""
    checkpoint_dirs = glob.glob(os.path.join(fold_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directory found in {fold_dir}")
    # Typically, the last saved checkpoint is the best one due to 'load_best_model_at_end=True'
    # A more robust way might be to parse trainer_state.json, but this is usually sufficient.
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint

# =================================================================================
# Cell 4: Main K-Fold Ensemble Evaluation Script
# =================================================================================

def evaluate_kfold_ensemble():
    """Loads all k-fold models, gets averaged predictions, and evaluates on the test set."""
    print("🚀 Starting Evaluation of K-Fold Ensemble on the Test Set...")
    NUM_LABELS = 7

    # 1. Load Tokenizer, Test Data, and Training Data for Preprocessing
    print("\n--- 1. Loading tokenizer and datasets ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    test_df = load_and_prepare_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    full_train_df = load_and_prepare_data(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)

    # 2. Preprocess Labels using the FULL training set
    print("\n--- 2. Preprocessing labels for evaluation ---")
    all_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'Z']
    mlb = MultiLabelBinarizer(classes=all_labels).fit(process_label_strings(full_train_df['labels_str']))
    train_labels_binary = mlb.transform(process_label_strings(full_train_df['labels_str']))
    cooccurrence_prob = analyze_label_cooccurrence(train_labels_binary, all_labels)
    print(f"Built co-occurrence map from {len(full_train_df)} training samples.")

    # 3. Tokenize Test Data
    print("\n--- 3. Tokenizing the test set ---")
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    test_dataset = MentalQADataset(test_encodings)

    # 4. Perform Ensemble Prediction
    print(f"\n--- 4. Generating predictions from {N_SPLITS} models ---")
    all_logits = []

    for i in range(N_SPLITS):
        fold = i + 1
        try:
            fold_dir = f"{K_FOLD_MODELS_DIR}_fold_{fold}"
            checkpoint_dir = find_best_checkpoint(fold_dir)
            weights_path = os.path.join(checkpoint_dir, 'model.safetensors')
            print(f"🔄 Processing Fold {fold}/{N_SPLITS} from: {checkpoint_dir}")

            # Instantiate a new model for this fold
            model = ImprovedMultiLabelModel(model_name=BASE_MODEL_NAME, num_labels=NUM_LABELS)
            state_dict = load_file(weights_path, device=DEVICE.type)
            model.load_state_dict(state_dict, strict=False) # Use strict=False for flexibility
            model.to(DEVICE)
            model.eval() # Set model to evaluation mode

            trainer = Trainer(model=model)
            raw_predictions = trainer.predict(test_dataset)
            all_logits.append(raw_predictions.predictions)

        except Exception as e:
            print(f"❌ Could not process Fold {fold}. Error: {e}")
            continue

    if not all_logits:
        print("❌ No models were successfully loaded. Aborting evaluation.")
        return

    # 5. Average the Logits from All Models
    print("\n--- 5. Averaging predictions (ensembling) ---")
    ensembled_logits = np.mean(all_logits, axis=0)
    print(f"✅ Successfully ensembled predictions from {len(all_logits)} models.")


    # 6. Post-process Ensembled Predictions
    print("\n--- 6. Applying adaptive thresholding to ensembled predictions ---")
    # Using the same threshold from your previous script for consistency
    best_threshold = 0.2462205131750359
    print(f"Using base_threshold: {best_threshold:.4f}")
    predicted_labels_list = adaptive_threshold_prediction(ensembled_logits, all_labels, cooccurrence_prob, base_threshold=best_threshold)

    # 7. Evaluate Final Predictions
    print("\n--- 7. Final Evaluation on the Test Set ---")
    y_true_binary = mlb.transform(process_label_strings(test_df['labels_str']))
    y_pred_binary = mlb.transform(predicted_labels_list)
    weighted_f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)

    print("\n--- 🥁 Final Ensembled Test Set Results 🥁 ---")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Jaccard Score:     {jaccard:.4f}")
    print("------------------------------------\n")
    print("--- Per-Label Performance (Test Set) ---")
    print(classification_report(y_true_binary, y_pred_binary, target_names=all_labels, zero_division=0))

    # 8. Save Predictions
    test_df['Predicted_Labels'] = [",".join(p) for p in predicted_labels_list]
    prediction_output_path = os.path.join(RESULTS_DIR, "kfold_ensembled_test_predictions.tsv")
    test_df[['Predicted_Labels']].to_csv(prediction_output_path, sep='\t', header=False, index=False)
    print(f"💾 Test set predictions saved to: {prediction_output_path}")
    print("\n✅ Evaluation complete.")

# Run the evaluation function
evaluate_kfold_ensemble()
