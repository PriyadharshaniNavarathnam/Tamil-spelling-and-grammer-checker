# gui.py
import tkinter as tk
from tkinter import scrolledtext
from model_utils import load_models, predict

root = tk.Tk()
root.title("Tamil Grammar Checker")
model, vectorizer, encoder = load_models()

def check_grammar():
    input_text = input_text_box.get("1.0", tk.END).strip()
    if input_text:
        corrected_text = predict(input_text, model, vectorizer, encoder)
        result_text_box.config(state='normal')
        result_text_box.delete('1.0', tk.END)
        result_text_box.insert(tk.END, corrected_text)
        result_text_box.config(state='disabled')

input_text_box = scrolledtext.ScrolledText(root, height=10, width=50)
input_text_box.pack(pady=10)

check_button = tk.Button(root, text="Check Grammar", command=check_grammar)
check_button.pack(pady=5)

result_text_box = scrolledtext.ScrolledText(root, height=10, width=50)
result_text_box.pack(pady=10)
result_text_box.config(state='disabled')

root.mainloop()
