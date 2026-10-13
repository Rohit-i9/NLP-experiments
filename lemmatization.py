
import nltk
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import Word

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# sentence
text = "Artificial Intelligence is changing the world. Python makes AI development easier!"

print("Original Sentence:")
print(text)

# Tokenize the sentence
words = word_tokenize(text)

# 1. NLTK WordNet Lemmatizer with POS tagging
lemmatizer = WordNetLemmatizer()
tagged_words = nltk.pos_tag(words)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

nltk_result = [
    lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    if word.isalpha() else word
    for word, tag in tagged_words
]

print("\n1. NLTK Lemmatization:")
print(nltk_result)

# 2. TextBlob Lemmatization
blob = TextBlob(text)
textblob_result = [
    Word(word).lemmatize(pos='v')
    for word in blob.words
]

print("\n2. TextBlob Lemmatization (verb mode):")
print(textblob_result)

# 3. spaCy Lemmatization
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_result = [token.lemma_ for token in doc]

print("\n3. spaCy Lemmatization:")
print(spacy_result)
