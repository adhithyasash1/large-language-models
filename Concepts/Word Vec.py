# Generating Word Vectors with Word2Vec

from gensim.models import Word2Vec
sentences = [['I', 'love', 'natural', 'language', 'processing'],
             ['Word', 'embeddings', 'convert', 'text', 'into', 'numbers']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)
word_vector = model.wv['natural']

# Exploring Word Similarities

similar_words = model.wv.most_similar('language', topn=5)

# Demonstrating Vector Arithmetic for Analogies

result_vector = model.wv['king'] - model.wv['man'] + model.wv['woman']
most_similar = model.wv.similar_by_vector(result_vector, topn=1)
