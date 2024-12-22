import json
import re
from nltk.translate.bleu_score import sentence_bleu
import tkinter as tk
from tkinter import scrolledtext

class GrammarChecker:
    def __init__(self, svo_rules_path, pronoun_rules_path):
        self.svo_rules = self.load_rules(svo_rules_path)
        self.pronoun_rules = self.load_rules(pronoun_rules_path)

    @staticmethod
    def load_rules(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Failed to load rules from {filepath}: File not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Failed to load rules from {filepath}: {e}")
            return {}

    def correct_svo_order(self, sentence):
        for rule in self.svo_rules.get("svo_order_rules", []):
            pattern = re.compile(rule["pattern"], re.IGNORECASE)
            if pattern.search(sentence):
                return rule["correction"], "SVO Order Error"
        return sentence, None

    def correct_pronoun_usage(self, sentence):
        original_sentence = sentence
        for incorrect_pronoun, correct_pronoun in self.pronoun_rules.get("pronoun_mapping", {}).items():
            pattern = re.compile(r'\b' + re.escape(incorrect_pronoun) + r'\b', re.IGNORECASE)
            sentence = pattern.sub(correct_pronoun, sentence)
        if sentence != original_sentence:
            return sentence, "Pronoun Usage Error"
        return sentence, None

    def check_grammar(self, sentence):
        corrected_sentence, svo_error = self.correct_svo_order(sentence)
        if svo_error:
            return corrected_sentence, svo_error
        corrected_sentence, pronoun_error = self.correct_pronoun_usage(corrected_sentence)
        if pronoun_error:
            return corrected_sentence, pronoun_error
        return corrected_sentence, None

def gui_display_results(test_sentences, reference_sentences, grammar_checker):
    root = tk.Tk()
    root.title("Grammar Checker Results")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30, font=("Times New Roman", 12))
    text_area.grid(column=0, pady=10, padx=10)

    for i, sentence in enumerate(test_sentences):
        corrected_sentence, error_type = grammar_checker.check_grammar(sentence)
        reference_sentence = reference_sentences[i]
        bleu_score = sentence_bleu([list(reference_sentence)], list(corrected_sentence))
        
        result_text = f"Sentence {i+1}:\nOriginal: {sentence}\nCorrected: {corrected_sentence}\nReference: {reference_sentence}\nError Type: {error_type or 'No errors detected.'}\nBLEU Score: {bleu_score:.4f}\n\n"
        text_area.insert(tk.END, result_text)

    root.mainloop()

# Specify the actual JSON files containing rules
svo_rules_path = "svo_order_rules.json"
pronoun_rules_path = "pronoun_rules.json"

# Initialize the GrammarChecker with paths to your rule files
grammar_checker = GrammarChecker(svo_rules_path, pronoun_rules_path)

# Example test and reference sentences to evaluate
test_sentences = [
    "அவர் மகளிடம் பரிசு கொடுத்தாள்.",
    "அவள் கதையை கேட்டான்.",
    "அவனை அழைத்தாள் தோழன்.",
    "அவர் கைகளை தூக்கியாள்.",
    "அவனை பார்த்தான் சிறுவர்கள்.",
    "அவர் கதையை கூறினாள்.",
    "அவள் மகனை அழைத்தான்.",
    "சிறுவர்களை அழைத்தாள் ஆசிரியர் வகுப்பில்.",
    "தோழன் பார்வையை மறைத்தான் மரம் தோட்டத்தில்.",
    "மெய்மறந்தார் பயணி காட்டில் அழகை கண்டது."
]

reference_sentences = [
    "அவர் மகளிடம் பரிசு கொடுத்தார்.",
    "அவள் கதையை கேட்டாள்.",
    "அவளை அழைத்தான் தோழன்.",
    "அவர் கைகளை தூக்கியார்.",
    "அவளை பார்த்தார்கள் சிறுவர்கள்.",
    "அவர் கதையை கூறினார்.",
    "அவள் மகனை அழைத்தாள்.",
    "ஆசிரியர் சிறுவர்களை வகுப்பில் அழைத்தார்.",
    "தோட்டத்தில் மரம் தோழனின் பார்வையை மறைத்தது.",
    "பயணி காட்டில் அழகை கண்டதும் மெய்மறந்தார்."
]

# Launch the GUI to display the results
gui_display_results(test_sentences, reference_sentences, grammar_checker)
