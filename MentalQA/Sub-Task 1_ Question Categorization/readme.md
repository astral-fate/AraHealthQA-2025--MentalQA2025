
Using device: cuda
Starting IMPROVED Multi-Label Classification with 'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment'...

--- Loading Data ---
Using all 350 examples for training.

--- Preprocessing Labels ---
Discovered 7 unique labels: ['A', 'B', 'C', 'D', 'E', 'F', 'Z']
Label processing complete.

--- Analyzing Label Co-occurrence ---
Found 14 strong label co-occurrence patterns

--- Tokenizing Text ---

--- Initializing Improved Model ---
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment and are newly initialized because the shapes did not match:
- classifier.weight: found shape torch.Size([3, 768]) in the checkpoint and torch.Size([7, 768]) in the model instantiated
- classifier.bias: found shape torch.Size([3]) in the checkpoint and torch.Size([7]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

--- Starting Enhanced Fine-Tuning ---
 [330/330 05:00, Epoch 15/15]
Step	Training Loss
20	0.318900
40	0.225500
60	0.195300
80	0.169200
100	0.134500
120	0.102500
140	0.067000
160	0.043300
180	0.027600
200	0.018200
220	0.013800
240	0.010600
260	0.008200
280	0.007400
300	0.006900
320	0.006500
Fine-tuning complete.

--- Generating Predictions for All Samples ---
Generated 350 predictions for submission
Predictions saved to '/content/drive/MyDrive/AraHealthQA/MentalQA/Task1/output/predictions_improved_camelbert.tsv'

--- Training Data Performance Analysis ---
Weighted F1 Score on training data: 0.9935

--- Per-Label Performance ---
              precision    recall  f1-score   support

           A       0.98      1.00      0.99       197
           B       0.99      1.00      1.00       203
           C       1.00      1.00      1.00        22
           D       0.98      1.00      0.99        80
           E       1.00      1.00      1.00        87
           F       1.00      1.00      1.00        14
           Z       1.00      1.00      1.00         6

   micro avg       0.99      1.00      0.99       609
   macro avg       0.99      1.00      1.00       609
weighted avg       0.99      1.00      0.99       609
 samples avg       0.99      1.00      0.99       609
