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
