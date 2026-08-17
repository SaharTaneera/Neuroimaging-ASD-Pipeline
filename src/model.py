# src/models.py
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_svm_classifier(kernel='poly', degree=3, C=1.0):
    """
    Initializes and returns a Support Vector Classifier optimized for fMRI connectomes.
    """
    model = SVC(kernel=kernel, degree=degree, C=C, probability=True, random_state=42)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model and returns accuracy, classification report, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, report, cm
