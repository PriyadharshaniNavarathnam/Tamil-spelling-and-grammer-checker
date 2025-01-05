import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

def load_data(filepath):
    data = pd.read_excel(filepath)
    print(f"Dataset size: {data.shape}")
    print(data.head())
    return data['Original Sentence'].values, data['Error Type'].values

def preprocess_data(sentences):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer

def encode_labels(labels):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y, encoder

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
    model.fit(X_train, y_train)
    
    # Evaluate on both training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], marker='o')
    plt.title('Model Accuracy: Train vs Test')
    plt.ylabel('Accuracy')
    plt.show()
    
    return model, X_train, X_test, y_train, y_test

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Plotting feature importance
    feature_importances = model.estimators_[0].feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

    return model, X_train, X_test, y_train, y_test

def correct_sentence(sentence, error_type):
    # This is a placeholder function to demonstrate correction based on error type.
    corrections = {
        "SVO Order Error": sentence[::-1],  # Reverse the sentence as a dummy correction
        "Pronoun Usage Error": sentence.replace("அவள்", "அவன்")
    }
    return corrections.get(error_type, sentence)  # Return the corrected sentence if error type is known, else return original

def test_sentences(model, encoder, sentences, vectorizer):
    transformed = vectorizer.transform(sentences)
    predictions = model.predict(transformed)
    predicted_labels = encoder.inverse_transform(predictions)
    
    # Open a file to write the results
    with open('grammar_check_results.txt', 'w', encoding='utf-8') as file:
        for sentence, prediction in zip(sentences, predicted_labels):
            corrected_sentence = correct_sentence(sentence, prediction)
            output = (f"Sentence: '{sentence}'\n"
                      f"Predicted Error: '{prediction}'\n"
                      f"Corrected: '{corrected_sentence}'\n\n")
            file.write(output)
            print(output)  # Optionally, you can also print the output

def main():
    filepath = r'D:\7th_Semester_FoE_UoJ\EC9640_Artificial Intelligence\Project\GrammarChecker\MachineLearning\DataSet\TamilDatasetGrammar.xlsx'
    sentences, labels = load_data(filepath)
    X, vectorizer = preprocess_data(sentences)
    y, encoder = encode_labels(labels)
    model, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    test_sentences_list = ["அவர் கதையை கேட்டாள்.", "அவள் மகனை அழைத்தான்.", "மாணவர்கள் பாடல்களை பாடினான்."]
    test_sentences(model, encoder, test_sentences_list, vectorizer)
    joblib.dump(model, 'tamil_grammar_checker_model.pkl')
    joblib.dump(vectorizer, 'tamil_vectorizer.pkl')
    joblib.dump(encoder, 'tamil_label_encoder.pkl')

if __name__ == '__main__':
    main()
