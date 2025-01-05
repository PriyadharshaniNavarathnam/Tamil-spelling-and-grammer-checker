from data_processing import load_data, preprocess_data, encode_labels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, vectorizer, encoder):
    joblib.dump(model, 'tamil_grammar_checker_model.pkl')
    joblib.dump(vectorizer, 'tamil_vectorizer.pkl')
    joblib.dump(encoder, 'tamil_label_encoder.pkl')

if __name__ == "__main__":
    filepath = r'D:\7th_Semester_FoE_UoJ\EC9640_Artificial Intelligence\Project\GrammarChecker\MachineLearning\DataSet\TamilDatasetGrammar.xlsx '
    sentences, labels, corrections = load_data(filepath)
    X, vectorizer = preprocess_data(sentences)
    y, encoder = encode_labels(labels)
    model, X_test, y_test = train_model(X, y)
    save_model(model, vectorizer, encoder)
