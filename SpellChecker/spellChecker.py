from symspellpy import SymSpell, Verbosity
from nltk import bigrams, FreqDist
from nltk.tokenize import word_tokenize
import nltk


def initialize_spell_checker(dictionary_path):
    """
    Initialize SymSpell with the provided dictionary file.

    Args:
        dictionary_path (str): Path to the dictionary file.

    Returns:
        SymSpell: Initialized SymSpell object.
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    try:
        with open(dictionary_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    sym_spell.create_dictionary_entry(parts[0], int(parts[1]))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")
    return sym_spell


def levenshtein_distance(word1, word2):
    """
    Calculate the Levenshtein distance between two words.

    Args:
        word1 (str): First word.
        word2 (str): Second word.

    Returns:
        int: The Levenshtein distance.
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def build_bigram_model(text_corpus):
    """
    Build a bigram language model from a text corpus.

    Args:
        text_corpus (str): The input text corpus.

    Returns:
        FreqDist: A frequency distribution of bigrams.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading 'punkt' tokenizer...")
        nltk.download("punkt")

    if not text_corpus.strip():
        raise ValueError("Text corpus is empty or invalid.")

    tokens = word_tokenize(text_corpus.lower())
    bigram_list = list(bigrams(tokens))
    return FreqDist(bigram_list)


def spell_check_with_levenshtein(text, sym_spell):
    """
    Perform spell checking with Levenshtein distance ranking.

    Args:
        text (str): Input text to spell check.
        sym_spell (SymSpell): Initialized SymSpell object.

    Returns:
        tuple: Corrected text, list of corrections made, and a dictionary of suggestions.
    """
    if not text.strip():
        return "", [], {}

    corrected_text = []
    corrections = []
    suggestions_dict = {}

    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            ranked_suggestions = sorted(
                suggestions, key=lambda s: levenshtein_distance(word, s.term)
            )
            corrected_word = ranked_suggestions[0].term

            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in ranked_suggestions[:3]]

            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)

    return " ".join(corrected_text), corrections, suggestions_dict


def spell_check_with_bigram(text, sym_spell, bigram_model):
    """
    Perform spell checking with contextual refinement using a bigram model.

    Args:
        text (str): Input text to spell check.
        sym_spell (SymSpell): Initialized SymSpell object.
        bigram_model (FreqDist): Bigram language model built from a corpus.

    Returns:
        tuple: Corrected text, list of corrections made, and a dictionary of suggestions.
    """
    if not text.strip():
        return "", [], {}

    words = text.split()
    corrected_text = []
    corrections = []
    suggestions_dict = {}

    for i, word in enumerate(words):
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            ranked_suggestions = sorted(
                suggestions, key=lambda s: levenshtein_distance(word, s.term)
            )
            corrected_word = ranked_suggestions[0].term

            if i > 0 and len(ranked_suggestions) > 1:
                prev_word = corrected_text[-1]
                bigram_candidates = [(prev_word, s.term) for s in ranked_suggestions]
                bigram_ranked = sorted(
                    bigram_candidates, key=lambda b: bigram_model.freq(b), reverse=True
                )

                if bigram_ranked and bigram_model.freq(bigram_ranked[0]) > 0:
                    corrected_word = bigram_ranked[0][1]

            corrected_text.append(corrected_word)
            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in ranked_suggestions[:3]]
        else:
            corrected_text.append(word)

    return " ".join(corrected_text), corrections, suggestions_dict


def spell_check_combined(text, sym_spell, bigram_model):
    """
    Perform spell checking by integrating a bigram model and Levenshtein distance.

    Args:
        text (str): Input text to spell check.
        sym_spell (SymSpell): Initialized SymSpell object.
        bigram_model (FreqDist): Bigram language model built from a corpus.

    Returns:
        tuple: Corrected text, list of corrections made, and dictionary of suggestions.
    """
    if not text.strip():
        return "", [], {}

    words = text.split()
    corrected_text = []
    corrections = []
    suggestions_dict = {}

    for i, word in enumerate(words):
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            ranked_suggestions = sorted(
                suggestions, key=lambda s: levenshtein_distance(word, s.term)
            )
            corrected_word = ranked_suggestions[0].term

            if i > 0 and len(ranked_suggestions) > 1:
                prev_word = corrected_text[-1]
                bigram_candidates = [(prev_word, s.term) for s in ranked_suggestions]
                bigram_ranked = sorted(
                    bigram_candidates, key=lambda b: bigram_model.freq(b), reverse=True
                )

                if bigram_ranked and bigram_model.freq(bigram_ranked[0]) > 0:
                    corrected_word = bigram_ranked[0][1]

            corrected_text.append(corrected_word)
            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in ranked_suggestions[:3]]
        else:
            corrected_text.append(word)

    return " ".join(corrected_text), corrections, suggestions_dict
