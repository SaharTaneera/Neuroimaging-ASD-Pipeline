os.environ["OMP_NUM_THREADS"] = "1"

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.preprocessing import load_abide_dataset, extract_features

def main():
    data_dir = "./ABIDE_data"
    direct_dir, labels_file = load_abide_dataset(data_dir=data_dir, n_subjects=100)
    
    X, y = extract_features(direct_dir, labels_file, max_files=100)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Support Vector Classifier...")
    model = SVC(kernel='poly', degree=3, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 16})
    plt.title('ABIDE fMRI Classification - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('abide_confusion_matrix.png')
    plt.close()

    joblib.dump(model, 'abide_svm_model.pkl')
    print("Pipeline completed successfully. Model saved to abide_svm_model.pkl")

if __name__ == "__main__":
    main()
