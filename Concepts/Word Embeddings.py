''' gpt '''

# Install the gensim library if you haven't already
# pip install gensim

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # Download the punkt tokenizer

# Sample text corpus
corpus = [
    "Word embeddings are a type of word representation",
    "They are used to capture semantic relationships",
    "Word2Vec is a popular method for creating word embeddings",
    "Gensim is a Python library for topic modeling and document similarity analysis",
]

# Tokenize the corpus into words
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model (optional)
model.save("word2vec.model")

# Get the vector representation of a word
vector = model.wv['word']
print(f"Vector representation of 'word': {vector}")

# Find similar words
similar_words = model.wv.most_similar('word', topn=3)
print(f"Words similar to 'word': {similar_words}")




''' Bard '''

import gensim.downloader as api
from gensim.models import Word2Vec

# Load a pre-trained word embedding model (e.g., Word2Vec trained on Google News)
model = api.load('word2vec-google-news-300')

# Explore word similarities
print(model.most_similar('king'))  # Output: [('queen', 0.75460547), ('prince', 0.7354554), ...]

# Get the vector representation of a word
word_vector = model['computer']
print(word_vector)  # Output: [-0.0054854, 0.065854, 0.00580754, ...]

# Train your own Word2Vec model on a specific dataset
sentences = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'sentence']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Save and load your trained model
model.save('my_word2vec_model')
new_model = Word2Vec.load('my_word2vec_model')
