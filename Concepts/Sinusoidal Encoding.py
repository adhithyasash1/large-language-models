import numpy as np

# Given word embedding x
x = np.array([0.1, 0, 0.23, 0.4, -0.75, 1])

# Position of the word in the sentence
pos = 4  # Position indexing starts from 1 in this context

# Dimensionality of the model, d_model (number of components in the embedding)
d_model = len(x)

# Calculating position embedding PE for each dimension of the word embedding
PE = np.array([np.sin(pos / (10000 ** (2 * i / d_model))) if i % 2 == 0 else np.cos(pos / (10000 ** ((2 * i - 1) / d_model))) for i in range(d_model)])

# Adding the position embedding PE to the word embedding x
h = x + PE

# Calculate the sum of the elements in h
sum_h = np.sum(h)

sum_h
