from spellChecker import levenshtein_distance
from bigram_spell_checker import build_bigram_model, spell_check_with_bigram

def spell_check_combined(text, sym_spell, bigram_model):
    if not text.strip():
        return "", [], {}

    words = text.split()
    corrected_text = []
    corrections = []
    suggestions_dict = {}

    for i, word in enumerate(words):
        corrected_text.append(word)  # Initialize with the original word
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            ranked_suggestions = sorted(suggestions, key=lambda s: levenshtein_distance(word, s.term))
            corrected_word = ranked_suggestions[0].term

            if i > 0:
                prev_word = corrected_text[-1]
                bigram_candidates = [(prev_word, s.term) for s in ranked_suggestions]
                bigram_ranked = sorted(bigram_candidates, key=lambda b: bigram_model.freq(b), reverse=True)

                if bigram_ranked and bigram_model.freq(bigram_ranked[0]) > 0:
                    corrected_word = bigram_ranked[0][1]

            corrected_text[-1] = corrected_word  # Update the word
            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in ranked_suggestions[:3]]

    return ' '.join(corrected_text), corrections, suggestions_dict
