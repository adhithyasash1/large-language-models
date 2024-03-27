import re, collections

# Calculate the frequency of pairs of symbols
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

# Merge the most frequent pair in the vocabulary
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, freq in v_in.items():
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = freq
    return v_out

# Perform BPE merges
def get_bpe_vocab(vocab, num_merges):
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

# Tokenize new text using the BPE vocabulary
def tokenize_bpe(text, vocab):
    words = text.split()
    tokens = []
    for word in words:
        word = ' '.join(list(word)) + ' </w>'
        for subword in vocab:
            while subword in word:
                tokens.append(subword)
                word = word.replace(subword, '', 1)
    return tokens

# Initial vocabulary setup with frequencies
vocab = {
    'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3
}

num_merges = 10
vocab = get_bpe_vocab(vocab, num_merges)

# Example text
text = "lower"

# Tokenizing new text
tokens = tokenize_bpe(text, vocab)
print("Tokens:", tokens)


# Assume `get_stats`, `merge_vocab`, and initial `vocab` setup from previous BPE example

def train_wordpiece(vocab, num_merges):
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

# Tokenize using the WordPiece model
def tokenize_wordpiece(text, vocab):
    # Split text into words, then into characters, and append '</w>'
    tokens = []
    for word in text.split():
        word_tokens = [' '.join(word) + ' </w>']
        for subword in sorted(vocab, key=len, reverse=True):  # Longest match first
            new_word_tokens = []
            for token in word_tokens:
                if subword in token:
                    new_word_tokens.extend(token.split(subword))
                else:
                    new_word_tokens.append(token)
            word_tokens = new_word_tokens
        tokens.extend(word_tokens)
    return [token for token in tokens if token.strip()]

# Example usage
vocab = train_wordpiece(vocab, num_merges)  # Assuming `vocab` is defined as before
text = "lower newest"
wordpiece_tokens = tokenize_wordpiece(text, vocab)
print("WordPiece Tokens:", wordpiece_tokens)
