# ecg_lancet_classification
1 Article

Attia, Zachi I., Peter A. Noseworthy, Francisco Lopez-Jimenez, Samuel J. Asirvatham, Abhishek J. Deshmukh, Bernard J. Gersh, Rickey E. Carter et al. "An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction." The Lancet 394, no. 10201 (2019): 861-867.

2 Overview

This code package for ECG classification includes modules for:
1) Model training, validation, and testing
2) Model loading and inference evaluation
3) Test performance calculation and metrics reporting
   

3 Implementation Steps

Step 1: Download the code package and extract the archive.

Step 2: Extract the dataset archive (records100.zip).

Step 3: Train the model by running attia_ecg_training_v1.py.

#python attia_ecg_training_v2.py \
--demograph-path ~/projects/04cv/data/ptb/ptb_scp_diag_simplified.csv \
--data-path ~/projects/04cv/data/ptb/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 \
--batch-size 100 \
--max-epochs 100 \
--learning-rate 1e-3 \
--weight-decay 0.0 \
--patience 8 \
--device cuda:2 \
--checkpoint-dir ~/projects/04cv/code/attia_checkpoints \
--result-path ~/projects/04cv/code/attia_kfold_results.pt \
--excel-result-path ~/projects/04cv/code/attia_kfold_testing_results.xlsx \
--positive-label 1

Step 4: Load the trained model and run testing using attia_ecg_load_and_test_v1.py.

#python attia_ecg_load_and_test_v1.py \
    --device cuda:2 \
    --kfold-data-path ~/projects/04cv/code/attia_kfold_scheme.pt \
    --checkpoint-dir ~/projects/04cv/code/attia_checkpoints \
    --result-path ~/projects/04cv/code/attia_post_training_test_results.pt \
    --excel-result-path ~/projects/04cv/code/attia_post_training_test_results.xlsx \
    --positive-label 1 \
    --batch-size 100 \
    --num-workers 0

Step 5: Calculate and evaluate test performance metrics using attia_test_performance.py.

#python attia_ecg_load_and_test_v1.py \
    --device cuda:1 \
    --kfold-data-path ~/projects/04cv/code/attia_kfold_scheme.pt \
    --checkpoint-dir ~/projects/04cv/code/attia_checkpoints \
    --result-path ~/projects/04cv/code/attia_post_training_test_results.pt \
    --excel-result-path ~/projects/04cv/code/attia_post_training_test_results.xlsx \
    --positive-label 1 \
    --batch-size 100 \
    --num-workers 0
