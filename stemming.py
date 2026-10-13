
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import RegexpStemmer

# Same sentence as the tokenization practical
text = "Artificial Intelligence is changing the world. Python makes AI development easier!"

# Tokenize the sentence
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
words = word_tokenize(text)

print("Original Sentence:")
print(text)

# 1. Porter Stemmer
porter = PorterStemmer()
print("\n1. Porter Stemmer:")
print([porter.stem(word) for word in words])

# 2. Snowball Stemmer
snowball = SnowballStemmer("english")
print("\n2. Snowball Stemmer:")
print([snowball.stem(word) for word in words])

# 3. Lancaster Stemmer
lancaster = LancasterStemmer()
print("\n3. Lancaster Stemmer:")
print([lancaster.stem(word) for word in words])

# 4. Regexp Stemmer
# Removes common suffixes such as ing, ed, and ly
regexp = RegexpStemmer(r'ing$|ed$|ly$|s$', min=3)
print("\n4. Regexp Stemmer:")
print([regexp.stem(word) for word in words])
