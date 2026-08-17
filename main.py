# main.py
os.environ["OMP_NUM_THREADS"] = "1"

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from src.preprocessing import load_abide_dataset, extract_features
from src.models import get_svm_classifier, evaluate_model

def main():
    data_dir = "./ABIDE_data"
    direct_dir, labels_file = load_abide_dataset(data_dir=data_dir, n_subjects=100)
    
    # Extract features using preprocessing module
    X, y = extract_features(direct_dir, labels_file, max_files=100)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model from models.py
    print("Training Support Vector Classifier...")
    model = get_svm_classifier(kernel='poly', degree=3)
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 16})
    plt.title('ABIDE fMRI Classification - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('abide_confusion_matrix.png')
    plt.close()

    # Save model artifact
    joblib.dump(model, 'abide_svm_model.pkl')
    print("Pipeline completed successfully. Model saved to abide_svm_model.pkl")

if __name__ == "__main__":
    main()
