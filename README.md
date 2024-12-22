# Tamil Spell and Grammar Checker

This project develops an advanced Tamil Spell and Grammar Checker leveraging a combination of rule-based algorithms and machine learning techniques. It is designed to enhance writing accuracy by correcting common spelling and grammatical errors specific to the Tamil language.

## Project Components

### Spell Checker
- **Levenshtein Distance**: Utilizes this algorithm to identify and suggest the smallest edits needed for spelling corrections.
- **Bigram Analysis**: Analyzes pairs of letters to predict and suggest corrections based on common letter combinations in Tamil.

### Grammar Checker
- **SVO and Pronoun Correction**: Specifically targets errors related to Subject-Verb-Object order and incorrect pronoun usage, crucial for maintaining sentence integrity in Tamil.
- **Rule-Based Corrections**: Applies predefined rules to quickly identify and correct habitual tense errors and ensure subject-verb agreement.
- **Seq2Seq with LSTM**: A sequence-to-sequence model using LSTM neural networks to understand and generate grammatically correct Tamil sentences, capturing deeper linguistic structures and contexts.

## Development Approach

### Data Preparation
- **Spell Checker**: Utilizes a corpus derived from extensive Tamil text datasets to train the bigram model and set up the Levenshtein distance calculations.
- **Grammar Checker**: Employs a rule-based system to address specific grammatical rules and a deep learning model to learn from a broad range of sentence structures.


## Practical Applications

This tool is ideal for educational platforms, content creation in Tamil media, and assisting non-native speakers in writing correct Tamil. 
It also serves as a valuable resource for linguistic research in Dravidian languages.

## Contributors

- Keerthikan F.J (2020E070)
-  Priyadharshani N. (2020E120)



