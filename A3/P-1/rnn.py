#pip install nltk, gensim
#python -m nltk.downloader stopwords

import pandas as pd
import re
from nltk.corpus import stopwords
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from nltk.stem import WordNetLemmatizer, PorterStemmer
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Define paths
train_path = './NLP3/NLP3/train.csv'
test_path = './NLP3/NLP3/test.csv'

#read csv
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

#randomly shuffle train and select 10% of data for validation
train = train.sample(frac=1,random_state=42).reset_index(drop=True)
val = train[:int(0.1*train.shape[0])]
train = train[int(0.1*train.shape[0]):]


#drop index column
train = train.drop(columns=['index'])
val = val.drop(columns=['index'])
test = test.drop(columns=['index'])

#check if data is loaded correctly
print(train.head())
print("Number of rows in train data: ", train.shape[0])
print(test.head())
print("Number of rows in test data: ", test.shape[0])
print(val.head())
print("Number of rows in validation data: ", val.shape[0])


print("Column names: ", train.columns)

#reallign text
def reallign(text):
    return ['__'.join([text[j][i] for j in range(len(text))]) for i in range(len(text[0]))]

#preprocess text
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

#preprocess text
train['text'] = train['text'].apply(preprocess_text)
val['text'] = val['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

sentences = [text.split() for text in train['text']]

# Define hyperparameters
max_seq_length = 50
batch_size = 30
num_hidden_units = 10
n_layers = 3
dropout = 0.3
num_output_units = 4
epoch_count = 100

#pad or truncate sentences to max_seq_length
def pad_truncate_sentences(sentences, max_seq_length):
    for i in range(len(sentences)):
        if len(sentences[i]) > max_seq_length:
            sentences[i] = sentences[i][:max_seq_length]
        else:
            sentences[i] = sentences[i] + ['']*(max_seq_length - len(sentences[i]))
    return sentences

#pad or truncate sentences
train['text'] = pad_truncate_sentences(sentences, max_seq_length)
val['text'] = pad_truncate_sentences([text.split() for text in val['text']], max_seq_length)
test['text'] = pad_truncate_sentences([text.split() for text in test['text']], max_seq_length)

#validate if all sentences have same length using assert
assert all(len(sentence) == max_seq_length for sentence in train['text'])
assert all(len(sentence) == max_seq_length for sentence in val['text'])
assert all(len(sentence) == max_seq_length for sentence in test['text'])

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe.iloc[index]['text']
        label = torch.tensor(self.dataframe.iloc[index]['label'])
        return {'text': text, 'label': label}
    
test_df = test
train_df = train
val_df = val

# Create dataloaders
train = CustomDataset(train)
val = CustomDataset(val)
test = CustomDataset(test)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

# Define RNN model
class RNN(nn.Module):
    def __init__(self, batch_size, embedding_dim, word2vec_model,hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.word2vec_model = word2vec_model
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional


        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    #forward pass    
    def forward(self, text):
        text = reallign(text)
        val = []
        total = None
        for sentence in text:
            embedded = []
            for token in sentence.split("__"):
                if token in self.word2vec_model.wv:
                    tensor1 = torch.tensor(self.word2vec_model.wv[token])
                    embedded.append(tensor1.view(1, -1))
                else:
                    tensor1 = torch.zeros(self.embedding_dim)
                    embedded.append(tensor1.view(1, -1))

            vectors = torch.stack(embedded)
            output, _ = self.rnn(vectors)
            output = self.dropout(output)
            output = output.view(-1, output.shape[-1])
            output = torch.mean(output, dim=0)
            output = output.view(1, -1)
            val.append(self.activation(self.fc(output)))
        return torch.stack(val)

    #initialize weights    
    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
        
        #initialize weights of classifier layer
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)
              
    #loss function
    def loss_function(self, outputs, labels):
        outputs = outputs.view(-1, outputs.shape[-1])
        return F.cross_entropy(outputs, labels)


#initialize model
word2vec_model = Word2Vec(sentences, min_count=10, sg=1)
embedding_dim = word2vec_model.vector_size

print("Embedding Dimension: ", embedding_dim)

model = RNN(batch_size,embedding_dim, word2vec_model, num_hidden_units, num_output_units, n_layers, True, dropout)
model.init_weights()

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
best_accuracy = 0
best_model = None
best_epoch = None

for total_epoch in range(50, 150, 50):
    for epoch in range(total_epoch):
        model.train()
        loss_total = 0
        k = 0
        for batch in train_loader:
            text = batch['text']
            labels = batch['label']
            optimizer.zero_grad()
            outputs = model(text)
            loss = model.loss_function(outputs, labels)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            k += 1
        print("Epoch: ", epoch, "Batch Loss: ", loss_total)
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            text = batch['text']
            labels = batch['label']
            outputs = model(text)
            predictions.extend(torch.argmax(outputs, dim=2).tolist())
            true_labels.extend(labels.tolist())
    accuracy = accuracy_score(true_labels, predictions)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_epoch = total_epoch
    print("Validation Accuracy: ", accuracy)

model = best_model        
torch.save(model.state_dict(), 'rnn.pth')  
# model.load_state_dict(torch.load('rnn.pth'))

# Evaluate the model
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        text = batch['text']
        labels = batch['label']
        outputs = model(text)
        predictions.extend(torch.argmax(outputs, dim=2).tolist())
        true_labels.extend(labels.tolist())

#save predictions and true labels in csv
predict = [pred[0] for pred in predictions]
test_df['preds'] = predict
test_df['text'] = test_df['text'].apply(lambda x: ' '.join([i for i in x if i != '']))
test_df.to_csv("rnn_test.csv")


# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')
conf_matrix = confusion_matrix(true_labels, predictions)
classification_rep = classification_report(true_labels, predictions)

# Print metrics
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)
print("Confusion Matrix: ", conf_matrix)
print("Classification Report: ", classification_rep)

