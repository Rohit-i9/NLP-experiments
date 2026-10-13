import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Import required libraries
import nltk
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Input sentence
text = "Artificial Intelligence is changing the world. Python makes AI development easier!"

print("Original Sentence:")
print(text)

# 1. Tokenization using Python split()
print("\n1. Python split() Tokenization:")
print(text.split())

# 2. Word tokenization using NLTK
print("\n2. NLTK Word Tokenization:")
print(word_tokenize(text))

# 3. Sentence tokenization using NLTK
print("\n3. NLTK Sentence Tokenization:")
print(sent_tokenize(text))

# 4. Tokenization using TextBlob
print("\n4. TextBlob Word Tokenization:")
blob = TextBlob(text)
print(blob.words)

# 5. Tokenization using spaCy
print("\n5. spaCy Tokenization:")
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print([token.text for token in doc])
