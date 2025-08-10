# arabert optmized 

# hyperparameters

```
Optimization Finished!
Best trial F1 Score: 0.6307
Best hyperparameters found:
  - learning_rate: 5.273957732715589e-05
  - num_train_epochs: 13
  - weight_decay: 0.04131058607286182
  - focal_alpha: 0.9702303056621574
  - focal_gamma: 1.39543909126709
  - base_threshold: 0.20408644287720523


```
# dev result 
Final Weighted F1 Score on dev data: 0.5928

```
--- Final Per-Label Performance on Dev Set ---
              precision    recall  f1-score   support

           A       0.67      0.94      0.78        33
           B       0.48      0.92      0.63        24
           C       0.00      0.00      0.00         4
           D       0.50      0.50      0.50        12
           E       0.26      0.90      0.40        10
           F       0.00      0.00      0.00         2
           Z       0.00      0.00      0.00         1

   micro avg       0.48      0.79      0.59        86
   macro avg       0.27      0.47      0.33        86
weighted avg       0.49      0.79      0.59        86
 samples avg       0.50      0.85      0.59        86

```

# test result

Weighted F1 Score: 0.5429

```
--- Per-Label Performance (Test Set) ---
              precision    recall  f1-score   support

           A       0.65      0.81      0.72        84
           B       0.60      0.75      0.67        85
           C       0.00      0.00      0.00        10
           D       0.37      0.21      0.26        34
           E       0.41      0.37      0.39        38
           F       0.00      0.00      0.00         6
           Z       0.00      0.00      0.00         3

   micro avg       0.58      0.59      0.58       260
   macro avg       0.29      0.31      0.29       260
weighted avg       0.51      0.59      0.54       260
 samples avg       0.65      0.65      0.60       260


```

