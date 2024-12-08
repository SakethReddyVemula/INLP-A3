import pandas as pd
import re
import numpy as np
import torch
import os
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD

def parameters_check(params_file, NUM_ROWS, WINDOW_SIZE, DIMENSION):
    saved_params = torch.load(params_file)
    return saved_params['NUM_ROWS'] == NUM_ROWS and saved_params['WINDOW_SIZE'] == WINDOW_SIZE and saved_params['DIMENSION'] == DIMENSION

class Tokenizer:
    def __init__(self):
        pass

    def tokenize_corpus(self, text):
        # Sentence Tokenizer
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        tokenized_text = []

        for sentence in sentences:
            tokenized_sentence = self.tokenize_sentence(sentence)
            tokenized_text.append(tokenized_sentence)

        return tokenized_text
    
    def tokenize_sentence(self, sentence):
        # Replace "\" with space
        sentence = re.sub(r'\\', r' ', sentence)

        # Time
        sentence = re.sub(r'\d:\d{2} [AP]M', r'', sentence)

        # Mentions
        sentence = re.sub(r'@[_\w]+', r'', sentence)

        # Hashtags
        sentence = re.sub(r'#[_\w]+', r'', sentence)

        # Mail IDs
        sentence = re.sub(r'[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+', r'', sentence)

        # URLs
        sentence = re.sub(r'(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?', r'', sentence)

        # Numbers
        sentence = re.sub(r'\d+.\d+|\d+|-\d+|\+\d+|\.\d+', r'', sentence)

        # Punctuation
        sentence = re.sub(r'[^\w\s]', r'', sentence)

        # Mobile Numbers
        sentence = re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'', sentence)

        # lower case
        sentence = sentence.lower()

        # return sentence
        return sentence.split()

class SVD:
    def __init__(self, path_to_data_file, minimum_freq=0, NUM_ROWS=10000, WINDOW_SIZE=1, DIMENSION=100):
        self.NUM_ROWS = NUM_ROWS
        self.WINDOW_SIZE = WINDOW_SIZE
        self.DIMENSION = DIMENSION
        self.tokenized_corpus = self.tokenize_corpus(path_to_data_file, NUM_ROWS)
        self.word_vectors_dict = self.get_svd_word_vectors(self.tokenized_corpus, WINDOW_SIZE, DIMENSION)
        self.vocab = list(self.word_vectors_dict.keys())
        
    def tokenize_corpus(self, path_to_file, NUM_ROWS):
        corpus = self.load_csv_corpus(path_to_file, NUM_ROWS)
        tokenized_corpus = self.Tokenize_Corpus(corpus)
        tokenized_corpus = [np.array(tokenized_sentence) for tokenized_sentence in tokenized_corpus]
        return tokenized_corpus

    def remove_source(self, description):
        if " - " in description:
            return description.split(" - ", 1)[1].strip()
        else:
            return description
    
    def load_csv_corpus(self, path, num_rows):
        data = pd.read_csv(path)
        data["Description"] = data['Description'].apply(self.remove_source)
        text_data = data["Description"][:num_rows]
        text_data = text_data.dropna()
        text_data = ' '.join(text_data)
        corpus = text_data
        return corpus

    def Tokenize_Corpus(self, corpus):
        tokenizer = Tokenizer()
        return tokenizer.tokenize_corpus(corpus)

    def get_svd_word_vectors(self, tokenized_corpus, WINDOW_SIZE, DIMENSION):
        vocab = list(set([word for sentence in tokenized_corpus for word in sentence]))

        co_occurance_matrix = np.zeros((len(vocab), len(vocab)))
        word_to_index = {word: i for i, word in enumerate(vocab)}
        self.word_to_index = word_to_index

        for sentence in tokenized_corpus:
            for i, word in enumerate(sentence):
                for j in range(max(0, i - WINDOW_SIZE), min(len(sentence), i + WINDOW_SIZE + 1)):
                    if i != j:
                        co_occurance_matrix[word_to_index[word], word_to_index[sentence[j]]] += 1

        svd_matrix = TruncatedSVD(n_components=DIMENSION)
        word_vectors_svd = svd_matrix.fit_transform(co_occurance_matrix)
        norms = np.linalg.norm(word_vectors_svd, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        word_vectors_normalized = word_vectors_svd / norms
        word_vectors_dict = {}
        for i, word in enumerate(vocab):
            word_vectors_dict[word] = word_vectors_normalized[i]
        word_vectors_dict["<UNK>"] = np.mean(list(word_vectors_dict.values()), axis=0)

        return word_vectors_dict

    def get_vector(self, word):
        if word not in self.vocab:
            return self.word_vectors_dict["<UNK>"]
        return self.word_vectors_dict[word]

    def similarity_pair(self, word1, word2):
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Handle zero norms
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def similarity(self, word, descending=True):
        sim = np.array(list(map(lambda x: self.similarity_pair(word, x), self.vocab)))
        sim_list = np.vstack((sim, np.array(self.vocab))).T

        if descending:
            rnk = np.argsort(sim)[::-1]
        else:
            rnk = np.argsort(sim)

        sim_list = sim_list[rnk]
        return sim_list
    
def print_embed(word, word_vectors_dict, DIMENSION):
    if word in word_vectors_dict.keys():
        return word_vectors_dict[word]
    else:
        return word_vectors_dict["<UNK>"]

if __name__ == "__main__":
    NUM_ROWS = 10000
    WINDOW_SIZE = 4
    DIMENSION = 100

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # file_name = f'svd_{NUM_ROWS}_{WINDOW_SIZE}_{DIMENSION}.pt'
    file_name = f'svd-word-vectors.pt'
    params_dict = {'NUM_ROWS': NUM_ROWS, 'WINDOW_SIZE': WINDOW_SIZE, 'DIMENSION': DIMENSION}


    # Train word vectors
    svd_model = SVD("train.csv", NUM_ROWS=NUM_ROWS, WINDOW_SIZE=WINDOW_SIZE, DIMENSION=DIMENSION)
    word_vectors_dict = svd_model.word_vectors_dict
    params_dict['word_vectors_dict'] = word_vectors_dict
    params_dict['vocab'] = svd_model.vocab
    params_dict['word_to_index'] = svd_model.word_to_index
        
    # Save word vectors and parameters to file
    torch.save(params_dict, file_name)
    print(f"Word vectors and parameters saved to {file_name}")

    # print(svd_model.get_vector('saketh'))
    # word = 'king'.lower()
    # print("Word Vector:", print_embed(word, word_vectors_dict, DIMENSION))

    print(svd_model.similarity('the')[:20])