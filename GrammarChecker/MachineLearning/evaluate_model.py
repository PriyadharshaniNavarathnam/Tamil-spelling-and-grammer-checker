# evaluate_model.py
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model_utils import load_models

def load_test_data():
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    return X_test, y_test

def evaluate_model(X_test, y_test):
    model, vectorizer, encoder = load_models()
    if not model:
        print("Failed to load the model.")
        return

    X_test_transformed = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    y_pred_label = encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred_label)
    print(f"Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred_label, labels=encoder.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    print(classification_report(y_test, y_pred_label))

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    evaluate_model(X_test, y_test)
