from symspellpy import Verbosity
from spellChecker import levenshtein_distance


def spell_check_with_levenshtein(text, sym_spell):
    """
    Perform spell checking with Levenshtein distance ranking.

    Args:
        text (str): Input text to spell check.
        sym_spell (SymSpell): Initialized SymSpell object.

    Returns:
        tuple: Corrected text, list of corrections made, and a dictionary of suggestions.
    """
    # Handle empty input
    if not text.strip():
        return "", [], {}

    corrected_text = []  # List to hold the corrected text
    corrections = []  # List to track corrections made
    suggestions_dict = {}  # Dictionary to store suggestions for each corrected word

    for word in text.split():
        # Get suggestions for the current word using SymSpell
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            # Rank suggestions by Levenshtein distance
            ranked_suggestions = sorted(
                suggestions, key=lambda s: levenshtein_distance(word, s.term)
            )
            corrected_word = ranked_suggestions[0].term

            # Track corrections and suggestions
            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in ranked_suggestions[:3]]  # Top 3 suggestions

            # Append the corrected word
            corrected_text.append(corrected_word)
        else:
            # If no suggestions, retain the original word
            corrected_text.append(word)

    # Return the corrected text, corrections made, and top suggestions
    return ' '.join(corrected_text), corrections, suggestions_dict
