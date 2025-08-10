# hyper-parameters


```
--- Using Optimized Hyperparameters ---
- learning_rate: 3.26662135376377e-05
- num_train_epochs: 10
- weight_decay: 0.0199876722361212
- focal_alpha: 1.194492474673312
- focal_gamma: 2.8990426579607704
- base_threshold: 0.3434835813289709

```

# dev

Overall Weighted F1 Score across all folds: 0.6189


```


--- Overall Per-Label Performance (based on out-of-fold predictions) ---
              precision    recall  f1-score   support

           A       0.62      0.98      0.76       197
           B       0.60      0.98      0.75       203
           C       0.18      0.23      0.20        22
           D       0.26      0.72      0.39        80
           E       0.33      0.70      0.45        87
           F       0.04      0.07      0.05        14
           Z       0.00      0.00      0.00         6

   micro avg       0.47      0.85      0.61       609
   macro avg       0.29      0.53      0.37       609
weighted avg       0.49      0.85      0.62       609
 samples avg       0.50      0.89      0.60       609


```

# test

- Weighted F1 Score: 0.5368
- Jaccard Score:     0.3862


```

--- Per-Label Performance (Test Set) ---
              precision    recall  f1-score   support

           A       0.56      0.71      0.63        84
           B       0.58      0.84      0.68        85
           C       0.00      0.00      0.00        10
           D       0.26      0.79      0.39        34
           E       0.26      0.89      0.40        38
           F       0.01      0.17      0.03         6
           Z       0.04      0.67      0.07         3

   micro avg       0.33      0.75      0.45       260
   macro avg       0.24      0.58      0.31       260
weighted avg       0.44      0.75      0.54       260
 samples avg       0.33      0.75      0.44       260

💾 Test set predictions saved to: /content/drive/MyDrive/AraHealthQA/MentalQA/Task1/results/kfold_ensembled_test_predictions.tsv

✅ Evaluation complete.

```
