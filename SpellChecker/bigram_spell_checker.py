import nltk
from nltk import bigrams, FreqDist
from nltk.tokenize import word_tokenize
from symspellpy import Verbosity
from spellChecker import levenshtein_distance

def build_bigram_model(text_corpus):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' tokenizer...")
        nltk.download('punkt')

    if not text_corpus.strip():
        raise ValueError("Text corpus is empty or invalid.")

    tokens = word_tokenize(text_corpus.lower())
    bigram_list = list(bigrams(tokens))
    return FreqDist(bigram_list)

def spell_check_with_bigram(text, sym_spell, bigram_model):
    if not text.strip():
        return "", [], {}

    words = text.split()
    corrected_text = []
    corrections = []
    suggestions_dict = {}

    for i, word in enumerate(words):
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            ranked_suggestions = sorted(suggestions, key=lambda s: levenshtein_distance(word, s.term))
            corrected_word = ranked_suggestions[0].term

            if i > 0 and len(ranked_suggestions) > 1:
                prev_word = corrected_text[-1]
                bigram_candidates = [(prev_word, s.term) for s in ranked_suggestions]
                bigram_ranked = sorted(bigram_candidates, key=lambda b: bigram_model.freq(b), reverse=True)

                if bigram_ranked and bigram_model.freq(bigram_ranked[0]) > 0:
                    corrected_word = bigram_ranked[0][1]

            corrected_text.append(corrected_word)
            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in ranked_suggestions[:3]]
        else:
            corrected_text.append(word)

    return ' '.join(corrected_text), corrections, suggestions_dict
