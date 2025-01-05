import joblib

def load_models():
    try:
        model = joblib.load(r'D:\7th_Semester_FoE_UoJ\EC9640_Artificial Intelligence\Project\GrammarChecker\MachineLearning\tamil_grammar_checker_model.pkl')
        vectorizer = joblib.load(r'D:\7th_Semester_FoE_UoJ\EC9640_Artificial Intelligence\Project\GrammarChecker\MachineLearning\tamil_vectorizer.pkl')
        encoder = joblib.load(r'D:\7th_Semester_FoE_UoJ\EC9640_Artificial Intelligence\Project\GrammarChecker\MachineLearning\tamil_label_encoder.pkl')
        return model, vectorizer, encoder
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def predict(text, model, vectorizer, encoder):
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)
    predicted_label = encoder.inverse_transform(prediction)[0]
    return predicted_label
