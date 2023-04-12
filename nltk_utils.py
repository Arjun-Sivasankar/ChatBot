import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenised_sentence, all_words):
    """
    :param tokenised_sentence: ["Hello", "how", "are", "you"]
    :param all_words: ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']
    :return: (bow) [0, 1, 0, 1, 0, 0, 0]
    """
    tokenised_sentence = [stem(w) for w in tokenised_sentence]

    bow = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenised_sentence:
            bow[idx] = 1.0

    return bow

# a = "What kinds of items are there?"
# print(a)
#
# b = tokenize(a)
# print(b)
#
# st_words = [stem(w) for w in b]
# print(st_words)
#
# sent = ["Hello", "how", "are", "you"]
# allw = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']
# print(bag_of_words(sent, allw))
