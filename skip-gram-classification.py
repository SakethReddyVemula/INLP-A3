import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
import pandas as pd
from collections import Counter
import numpy as np
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

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
    
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<UNK>'
PADDING_LENGTH = 30
    
class MyDataset(Dataset):
    def __init__(self, data, vocab_to_index):
        self.sentences = [i[0] for i in data]
        self.labels = [i[1] for i in data]
        self.vocab2index = vocab_to_index

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence_indices = [self.vocab2index.get(token, self.vocab2index[UNKNOWN_TOKEN]) for token in self.sentences[index]]
        return torch.tensor(sentence_indices), torch.tensor(self.labels[index])
    
    def collate(self, batch):
        sentences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Convert tokens to indices for each sentence
        sentence_indices = [torch.tensor([token for token in sentence]) for sentence in sentences]
        padded_sequences = pad_sequence(sentence_indices, batch_first=True, padding_value=self.vocab2index[PAD_TOKEN])
        padded_labels = torch.tensor(labels)  # Convert labels to tensor
        # length = torch.LongTensor([len(sentence) for sentence in padded_sequences])
        length = torch.LongTensor([PADDING_LENGTH for sentence in padded_sequences])
        return padded_sequences, length, padded_labels

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(LSTMClassifier, self).__init__()
        pretrained_embeddings = torch.FloatTensor(pretrained_embeddings)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, input_lengths):
        x = self.embedding(x.long())
        packed_embedding = pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(packed_embedding)
        output, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # output = torch.mean(output, dim=1) # Line 1
        # output = self.linear(output) # Line 1
        output = self.linear(output[:, -1, :]) # Line 2
        # print(output)
        return output
    
def calculate_metrics(true_labels, predicted_labels, num_classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    correct_predictions = (np.array(true_labels) == np.array(predicted_labels)).astype(float)
    accuracy = correct_predictions.sum() / len(true_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)

    # Add numerical values inside the boxes
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(range(num_classes), rotation=45)
    ax.set_yticklabels(range(num_classes))
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    # plt.show()

    # Save the plot as an image
    plt.savefig('sg_4.png', dpi=300, bbox_inches='tight')

    return cm, accuracy, f1, precision, recall

if __name__ == "__main__":
    # Hyperparamers
    BATCH_SIZE = 64
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 100
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = Tokenizer()

    train_data = pd.read_csv('train.csv').head(10000)
    test_data = pd.read_csv('test.csv').head(1000)

    data_train = []
    for i, row in train_data.iterrows():
        data_train.append((tokenizer.tokenize_sentence(row['Description']), row['Class Index'] - 1))

    data_test = []
    for i, row in test_data.iterrows():
        data_test.append((tokenizer.tokenize_sentence(row['Description']), row['Class Index'] - 1))

    # params_dict = torch.load('sg_10000_4_100_5.pt')
    params_dict = torch.load('skip-gram-word-vectors.pt')
    word_vectors_dict = params_dict['word_vectors_dict']
    word_to_index = params_dict['word_to_index']
    word_to_index[UNKNOWN_TOKEN] = len(word_to_index)

    new_word_to_index = {}

    for i, word in enumerate(word_to_index):
        new_word_to_index[word] = i - 1
        if new_word_to_index[word] == -1:
            new_word_to_index[word] = len(word_to_index) - 1

    sorted_new_word_to_index = dict(sorted(new_word_to_index.items(), key=lambda item: item[1]))

    word_to_index = sorted_new_word_to_index
    
    # word_to_index[PAD_TOKEN] = len(word_to_index)
    
    pretrained_embeddings = np.zeros((len(word_to_index), EMBEDDING_DIM))
    for i, word in enumerate(word_to_index):
        # print(i, word)
        pretrained_embeddings[i] = word_vectors_dict.get(word, word_vectors_dict['<UNK>'])

    dataset_train = MyDataset(data_train, word_to_index)
    dataset_test = MyDataset(data_test, word_to_index)

    OUTPUT_DIM = len(set(dataset_train.labels))

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset_train.collate)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset_test.collate)

    model = LSTMClassifier(len(word_to_index), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, pretrained_embeddings).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        # model.train()
        for batch in dataloader_train:
            input_data, length, labels = batch
            input_data = input_data.to(device)
            # print("input_data:", input_data)
            # print("labels:", labels)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(input_data, length)
            loss = loss_fn(predictions, labels)
            # print(predictions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(dataloader_train)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

    torch.save(model, 'skip-gram-classification-model.pt')

    model.eval()  # Set the model to evaluation mode
    all_true_labels = []
    all_predicted_labels = []

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader_test:
            input_data, length, labels = batch
            input_data = input_data.to(device)
            labels = labels.to(device)
            predictions = model(input_data, length)
            _, predicted = torch.max(predictions, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.2%}')

    # Calculate metrics
    cm, accuracy, f1, precision, recall = calculate_metrics(all_true_labels, all_predicted_labels, OUTPUT_DIM)

    # Print results
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
