#pip install nltk, gensim
#python -m nltk.downloader stopwords

import pandas as pd
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import csv
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


train_path = './NLP3/NLP3/train.csv'
test_path = './NLP3/NLP3/test.csv'

#read csv
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

#randomly shuffle train and select 10% of data for validation
train = train.sample(frac=1).reset_index(drop=True)
val = train[:int(0.1*train.shape[0])]
train = train[int(0.1*train.shape[0]):]


#drop index column
train = train.drop(columns=['index'])
val = val.drop(columns=['index'])
test = test.drop(columns=['index'])


print(train.head())
print("Number of rows in train data: ", train.shape[0])
print(test.head())
print("Number of rows in test data: ", test.shape[0])
print(val.head())
print("Number of rows in validation data: ", val.shape[0])

#print column names
print(train.columns)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.replace('\\', ' ')
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]    
    preprocessed_text = ' '.join(filtered_words)

    # lemmatizer = WordNetLemmatizer()
    # preprocessed_text = ' '.join([lemmatizer.lemmatize(word) for word in preprocessed_text.split()])

    stemmer = PorterStemmer()
    preprocessed_text = ' '.join([stemmer.stem(word) for word in preprocessed_text.split()])
    return preprocessed_text

train['text'] = train['text'].apply(preprocess_text)
val['text'] = val['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

sentences = [text.split() for text in train['text']]
max_seq_length = 50


def pad_truncate_sentences(sentences, max_seq_length):
    for i in range(len(sentences)):
        if len(sentences[i]) > max_seq_length:
            sentences[i] = sentences[i][:max_seq_length]
        else:
            sentences[i] = sentences[i] + ['']*(max_seq_length - len(sentences[i]))
    return sentences


train['text'] = pad_truncate_sentences(sentences, max_seq_length)
val['text'] = pad_truncate_sentences([text.split() for text in val['text']], max_seq_length)
test['text'] = pad_truncate_sentences([text.split() for text in test['text']], max_seq_length)

#validate if all sentences have same length using assert
assert all(len(sentence) == max_seq_length for sentence in train['text'])
assert all(len(sentence) == max_seq_length for sentence in val['text'])
assert all(len(sentence) == max_seq_length for sentence in test['text'])

batch_size = 32
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]
    
train = PandasDataset(train)
val = PandasDataset(val)
test = PandasDataset(test)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


class RNN(nn.Module):
    def __init__(self, embedding_dim, word2vec_model,hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.wv.vectors))
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)
        
        #define forward function
        #pass avaerage of outputs of rnn for all tokens to classifier layer
        #apply activation function
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.activation(self.fc(hidden))
    
    def init_weights(self):
        #initialize weights of embedding layer
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        #initialize weights of rnn layer
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
        
        #initialize weights of classifier layer
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)
        
        #define count_parameters function
        #count number of trainable parameters in the model
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def loss_function(self, outputs, labels):
        #define loss function
        #use cross entropy loss
        return F.cross_entropy(outputs, labels)
    
    def back_propagation(self, loss):
        #define back propagation function
        #use adam optimizer
        optimizer = optim.Adam(self.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#initialize model
word2vec_model = Word2Vec(sentences, min_count=10, sg=1)
embedding_dim = word2vec_model.vector_size

model = RNN(embedding_dim, word2vec_model, 50, 2, 10, True, 0.5)
model.init_weights()
print(model.count_parameters())

for epoch in range(50):
    model.train()
    for batch in train_loader:
        
        text = batch['text']
        labels = batch['label']
        print("Saransh",text)
        outputs = model(text)
        loss = model.loss_function(outputs, labels)
        model.back_propagation(loss)
        
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            text = batch['text']
            labels = batch['label']
            outputs = model(text)
            loss = model.loss_function(outputs, labels)
            print("Epoch: ", epoch, "Validation Loss: ", loss.item())



