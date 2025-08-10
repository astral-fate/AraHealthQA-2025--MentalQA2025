## 1st attempt: The AraBERTv2 kfold ensemble used the same hyperparameters as the optimized single AraBERTv2 model for consistency
## fine-tuning mentalQA usinf arabert2 using k-folds ensemble 

# fine-tuning parameters

```
--- Using Optimized Hyperparameters ---
learning_rate: 5.273957732715589e-05
num_train_epochs: 13
weight_decay: 0.04131058607286182
focal_alpha: 0.9702303056621574
focal_gamma: 1.39543909126709
base_threshold: 0.20408644287720523

```


# dev result


Overall Weighted F1 Score across all folds: 0.6066

```
--- Overall Per-Label Performance (based on OOF predictions) ---
              precision    recall  f1-score   support

           A       0.59      0.99      0.74       197
           B       0.60      0.98      0.74       203
           C       0.16      0.23      0.19        22
           D       0.25      0.79      0.38        80
           E       0.29      0.91      0.43        87
           F       0.00      0.00      0.00        14
           Z       0.00      0.00      0.00         6

   micro avg       0.44      0.89      0.59       609
   macro avg       0.27      0.56      0.36       609
weighted avg       0.47      0.89      0.61       609
 samples avg       0.46      0.92      0.58       609
# test result


Using base_threshold: 0.2041

```

# test result 

```
--- 🥁 Final Ensembled Test Set Results 🥁 ---
- Weighted F1 Score: 0.2597
- Jaccard Score:     0.1940
------------------------------------

--- Per-Label Performance (Test Set) ---
              precision    recall  f1-score   support

           A       0.57      0.94      0.71        84
           B       0.00      0.00      0.00        85
           C       0.07      1.00      0.12        10
           D       0.18      0.18      0.18        34
           E       0.00      0.00      0.00        38
           F       0.05      1.00      0.09         6
           Z       0.02      1.00      0.04         3

   micro avg       0.17      0.40      0.24       260
   macro avg       0.13      0.59      0.16       260
weighted avg       0.21      0.40      0.26       260
 samples avg       0.17      0.37      0.23       260


```

# 2nd attempt; The AraBERTv2 kfold ensemble used the hyperparameters as the optimized single CAMERL-BERT model

# hyper-parameters 

```
--- Using Optimized Hyperparameters ---
learning_rate: 3.26662135376377e-05
num_train_epochs: 10
weight_decay: 0.0199876722361212
focal_alpha: 1.194492474673312

```

# dev reuslt

Overall Weighted F1 Score across all folds: 0.6183
# Results

- 🥁 Final Ensembled Test Set Results 🥁 ---
- Weighted F1 Score: 0.3283
- Jaccard Score:     0.2382

# notebook

https://colab.research.google.com/drive/19NClhPis--SpLjxllBNlOiQoWl22l77w
