# hyper prameters 


```
Best trial F1 Score: 0.6426
Best hyperparameters found:
  - learning_rate: 6.41632473330898e-05
  - num_train_epochs: 14
  - weight_decay: 0.0480197212055279
  - focal_alpha: 1.23204424578726
  - focal_gamma: 2.624023968262825
  - base_threshold: 0.2462205131750359

```
# dev results

Final Weighted F1 Score on dev data: 0.6194

```

--- Final Per-Label Performance on Dev Set ---
              precision    recall  f1-score   support

           A       0.67      1.00      0.80        33
           B       0.48      1.00      0.65        24
           C       0.29      0.50      0.36         4
           D       0.33      1.00      0.50        12
           E       0.25      0.70      0.37        10
           F       0.00      0.00      0.00         2
           Z       0.00      0.00      0.00         1

   micro avg       0.44      0.91      0.59        86
   macro avg       0.29      0.60      0.38        86
weighted avg       0.48      0.91      0.62        86
 samples avg       0.45      0.94      0.58        86

```
# test result

- Weighted F1 Score: 0.5972
- Jaccard Score:     0.4502

```


--- Per-Label Performance (Test Set) ---
              precision    recall  f1-score   support

           A       0.61      0.96      0.75        84
           B       0.56      0.98      0.72        85
           C       0.14      0.40      0.21        10
           D       0.25      0.91      0.39        34
           E       0.34      0.53      0.41        38
           F       0.07      0.33      0.11         6
           Z       0.00      0.00      0.00         3

   micro avg       0.42      0.85      0.57       260
   macro avg       0.28      0.59      0.37       260
weighted avg       0.47      0.85      0.60       260
 samples avg       0.44      0.88      0.56       260

```
