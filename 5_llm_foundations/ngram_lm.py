# Count matrix: Count of occurrences of next letter given current - bigram
# Probability matrix: Count matrix values / Row total
# Language model: Generate final probability
# Log probability to avoid underflow: Addition of probabilities
# Generative language model: 80% train, 10% validation, 10% test for small datasets

from collections import defaultdict

import numpy as np
import pandas as pd


def single_pass_trigram_count_matrix(corpus):
    """
    Creates the trigram count matrix from the input corpus in a single pass through the corpus.

    Args:
        corpus: Pre-processed and tokenized corpus.

    Returns:
        bigrams: list of all bigram prefixes, row index
        vocabulary: list of all found words, the column index
        count_matrix: pandas dataframe with bigram prefixes as rows,
                      vocabulary words as columns
                      and the counts of the bigram/word combinations (i.e. trigrams) as values
    """
    bigrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(dict)

    # go through the corpus once with a sliding window
    for i in range(len(corpus) - 3 + 1):
        # the sliding window starts at position i and contains 3 words
        trigram = tuple(corpus[i: i + 3])

        bigram = trigram[0: -1]
        if not bigram in bigrams:
            bigrams.append(bigram)

        last_word = trigram[-1]
        if not last_word in vocabulary:
            vocabulary.append(last_word)

        if (bigram, last_word) not in count_matrix_dict:
            count_matrix_dict[bigram, last_word] = 0

        count_matrix_dict[bigram, last_word] += 1

    # convert the count_matrix to np.array to fill in the blanks
    count_matrix = np.zeros((len(bigrams), len(vocabulary)))

    for trigram_key, trigam_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]), \
            vocabulary.index(trigram_key[1])] \
            = trigam_count

    # np.array to pandas dataframe conversion
    count_matrix = pd.DataFrame(count_matrix, index=bigrams, columns=vocabulary)
    return bigrams, vocabulary, count_matrix


corpus = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']

bigrams, vocabulary, count_matrix = single_pass_trigram_count_matrix(corpus)

print(count_matrix)

row_sums = count_matrix.sum(axis=1)
# divide each row by its sum
prob_matrix = count_matrix.div(row_sums, axis=0)

print(prob_matrix)

suggestions = {}

for word in vocabulary:
    probability = prob_matrix.loc[[('i', 'am')]][word]
    print(probability.values)
    suggestions[word] = probability.values[0]

final_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)

print(final_suggestions)
print("I am ________")
print("Option 1", final_suggestions[0][0])
print("Option 2", final_suggestions[1][0])
