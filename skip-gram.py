import re
import torch
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
import pandas as pd
import os

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

        # # Punctuation
        # sentence = re.sub(r'^[^\w\s\<\>]$', r'', sentence)
        # Punctuation
        sentence = re.sub(r'[^\w\s]', r'', sentence)

        # Mobile Numbers
        sentence = re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'', sentence)

        # lower case
        sentence = sentence.lower()

        # return sentence
        return sentence.split()

class Corpus:
    """
        Class for Corpus related tasks
    """
    def __init__(self, path_to_data_file, minimum_freq=0, NUM_ROWS=10):
        self.NUM_ROWS = NUM_ROWS
        self.tokenized_corpus = self.tokenize_corpus(path_to_data_file, NUM_ROWS) # [array(['abs', 'basd'], dtype='<U2'),array(['csd', 'ads'], dtype='<U2')]
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset_line(self.tokenized_corpus, minimum_freq) # self.data = [[1, 34, 13], [23, 43]]
        self.vocab_size = len(self.dictionary) - 1 # -1 for removing Unknown from the count
        self.negative_sample_table_w, self.negative_sample_table_p = self.create_negative_sample_table()
        # print("negative_sample_table_w:", self.negative_sample_table_w)
        # print("negative_sample_table_p:", self.negative_sample_table_p)

    def build_dataset_line(self, tokenized_corpus, minimum_freq):
        """
            Input: tokenized_corpus, min_cutoff_freq
            Output: data(indexed data: [[1, 2, 3], [34, 54]]), count, dictionary{'a': 1, 'b': 2}, rev_dictionary
        """
        count = [["<UNK>", -1]]
        words = np.concatenate(tokenized_corpus)
        count.extend(collections.Counter(words).most_common())
        count = np.array(count)
        count = count[(count[:, 1].astype(int) >= minimum_freq) | (count[:, 1].astype(int) == -1)]

        # dictionary
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        # data
        data = list()
        unk_count = 0
        for tokenized_sentence in tokenized_corpus:
            # print(tokenized_sentence)
            data_temp = list()
            for token in tokenized_sentence:
                if token in dictionary:
                    index = dictionary[token]
                else:
                    index = 0 # dictionary["<UNK>"]
                    unk_count += 1
                data_temp.append(index)
            if(len(data_temp) > 0):
                data.append(data_temp)
        count[0][1] = unk_count
        return data, count, dictionary, reverse_dictionary


    def tokenize_corpus(self, path_to_file, NUM_ROWS):
        """
            Driver function to tokenize the corpus and return a list of np.arrays
        """
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
        """
            Tokenize the corpus
        """
        tokenizer = Tokenizer()
        return tokenizer.tokenize_corpus(corpus)
    
    def create_negative_sample_table(self):
        """
            output: negative_sample_table_w: np.array([1, 2, 3, 4, 5])
                    negative_sample_table_p: np.array([0.6, 0.3, 0.06, 0.03, 0.01]) ==> unigram distribution raised to the power 3/4 and normalized
        """
        word_counts = self.count[1:, 1].astype(int) #index 0 is unk
        p = word_counts / word_counts.sum()
        p = np.power(p, 0.75)
        p /= p.sum()
        return np.array(list(range(1, len(self.dictionary)))), p
    
    def negative_sampling(self, num_negative_samples):
        """
        Samples negative examples from the unigram distribution raised to the 3/4 power.
        """
        negative_samples = np.random.choice(
            self.negative_sample_table_w,
            size=num_negative_samples,
            replace=True,
            p=self.negative_sample_table_p
        )
        return negative_samples
    

class word2vec(torch.nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_dim, negative_samples):
        super(word2vec, self).__init__()
        self.negative_samples = negative_samples
        self.row_idx = 0
        self.col_idx = 0
        self.batch_end = 0
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size + 1
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        if torch.cuda.is_available():
            self.embedding = self.embedding.cuda()

        # embedding initialization
        init_range = 0.5 / self.embedding_dim
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def generate_batch(self, corpus, window_size):
        """
            Generating batches of input and output pairs.
            x: an array of input words
            y: an array of target words
            neg: an array of negative samples
        """
        row_idx = self.row_idx
        col_idx = self.col_idx
        inputs = collections.deque()
        targets = collections.deque()
        neg_samples = collections.deque()
        i = 0

        while row_idx < len(corpus.data):
            data = corpus.data[row_idx]
            target = data[col_idx]
            length_of_sentence = len(data)

            start_idx = max(0, col_idx - window_size)
            end_idx = min(col_idx + window_size + 1, length_of_sentence)
            context = [data[c] for _, c in enumerate(range(start_idx, end_idx)) if c != col_idx]

            for input_word in context:
                inputs.append(input_word)
                targets.append(target)
                neg_samples.append(corpus.negative_sampling(self.negative_samples))

            col_idx += 1

            if col_idx == len(data):
                col_idx = 0
                row_idx += 1

        x = np.array(inputs)
        y = np.array(targets)
        neg = np.array(neg_samples)
        return x, y, neg

    def forward(self, batch, corpus=None):
        """
            Objective function: to discriminate between the target word and the negative samples, given the input word.
            Loss function: negative sampling loss
        """
        """
            batch[0]: Input embedding list: [23, 25, 27, 29]
            batch[1]: Target embedding list : [24, 24, 28, 28]
            batch[2]: Negative samples list for each input-target pair: [[19, 21, 17, 39, 24], [31, 20, 9, 28, 39], ...]
        """
        # positive
        input_emb = self.embedding(batch[0])
        output_emb = self.embedding(batch[1])

        score = torch.sum(torch.mul(input_emb, output_emb), dim=1)  # inner product
        log_target = F.logsigmoid(score)

        # negative
        output_emb_negative = self.embedding(batch[2].view(-1))
        neg_score = -1 * torch.sum(torch.mul(output_emb.view(-1, 1, self.embedding_dim), output_emb_negative.view(-1, self.negative_samples, self.embedding_dim)), dim=2)
        log_neg_sample = F.logsigmoid(neg_score)

        loss = -1 * (log_target.sum() + log_neg_sample.sum())
        return loss
            
class Use_model:
    def __init__(self, corpus, embedding_dim, window_size, batch_size, num_epochs, negative_samples=10, trace=False):
        self.corpus = corpus
        self.window_size = window_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.trace = trace
        self.model_path =  f"SGmodel_{self.embedding_dim}_{self.batch_size}_{num_epochs}.pt"
        self.model = word2vec(self.corpus.vocab_size, self.embedding_dim, self.negative_samples)

        # Check if model file exists
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)))
            print("Model loaded successfully.")

    def train(self, num_epochs=100, learning_rate=0.001):
        if os.path.exists(self.model_path):
            print("Model already exists. Skipping training.")
            return
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        x, y, neg = self.model.generate_batch(self.corpus, self.window_size)

        x = torch.LongTensor(x).to(device)
        y = torch.LongTensor(y).to(device)
        neg = torch.LongTensor(neg).to(device)
        epo_start_time = time.time()

        for epo in range(num_epochs):
            loss_val = 0
            dataset = torch.utils.data.TensorDataset(x, y, neg)
            batches = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for batch in batches:
                optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()
                loss_val += loss.data
                optimizer.step()

            if epo % 2 == 0:
                print(time.time() - epo_start_time)
                epo_start_time = time.time()
                print(f"Loss at epo {epo}: {loss_val / len(batches)}")

        # Save Model
        torch.save(self.model.state_dict(), f"SGmodel_{self.embedding_dim}_{self.batch_size}_{num_epochs}.pt")

    def get_vector(self, word):
        word_idx = self.corpus.dictionary[word]
        word_idx = torch.LongTensor([[word_idx]])

        if torch.cuda.is_available():
            word_idx = word_idx.to(device)
            vector = self.model.embedding(word_idx).view(-1).detach().cpu().numpy()
        else:
            vector = self.model.embedding(word_idx).view(-1).detach().numpy()
        return vector

    def similarity_pair(self, word1, word2):
        return np.dot(self.get_vector(word1), self.get_vector(word2)) / (np.linalg.norm(self.get_vector(word1)) * np.linalg.norm(self.get_vector(word2)))

    def similarity(self, word, descending=True):
        words = np.array([x for x in self.corpus.dictionary.items()])
        sim = np.array(list(map(lambda x: self.similarity_pair(word, x[0]), words)))  # calculate similarity
        sim_list = np.vstack((sim, words[:, 0])).T

        if descending:
            rnk = np.argsort(sim)[::-1]
        else:
            rnk = np.argsort(sim)

        sim_list = sim_list[rnk]
        return sim_list
    
if __name__ == "__main__":
    start = time.time()
    # Hyperparameters
    NUM_ROWS = 10000
    WINDOW_SIZE = 4
    EMBEDDING_DIMS = 100
    BATCH_SIZE = 100
    NEGATIVE_SAMPLES = 5
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    # Check if CUDA is available
    filename = f'skip-gram-word-vectors.pt'
    params_dict = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    modelCorpus = Corpus("train.csv", 0, NUM_ROWS)
    word_to_index = modelCorpus.dictionary
    print(word_to_index)
    params_dict['word_to_index'] = word_to_index

    
    model = Use_model(modelCorpus, EMBEDDING_DIMS, WINDOW_SIZE, BATCH_SIZE, NUM_EPOCHS, negative_samples=NEGATIVE_SAMPLES, trace=True)
    model.train(num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    process_time = time.time() - start
    print("Total_Training_Time:", process_time)

    word_vectors_dict = {}  
    for word in word_to_index.keys():
        word_vectors_dict[word] = model.get_vector(word)

    # print(word_vectors_dict['<UNK>'])
    params_dict['word_vectors_dict'] = word_vectors_dict

    torch.save(params_dict, filename)
    print(f"Word vectors and parameters saved to {filename}")
    
    # print(model.get_vector("a"))
    # print(model.get_vector("<UNK>"))
    # print(model.similarity('olympic'.lower())[:20])
