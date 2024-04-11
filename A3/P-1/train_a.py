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
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer


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

    #perform lemmatization
    # lemmatizer = WordNetLemmatizer()
    # preprocessed_text = ' '.join([lemmatizer.lemmatize(word) for word in preprocessed_text.split()])

    #perform stemming
    stemmer = PorterStemmer()
    preprocessed_text = ' '.join([stemmer.stem(word) for word in preprocessed_text.split()])


    return preprocessed_text

#apply preprocessing to text column
train['text'] = train['text'].apply(preprocess_text)
val['text'] = val['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

sentences = [text.split() for text in train['text']]
word2vec_model = Word2Vec(sentences, min_count=10, sg=1)

def get_sentence_embeddings(text, model, embedding_dim):
    words = text.split()
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
    if len(embeddings) == 0:
        return np.zeros(embedding_dim)
    else:
        return np.mean(embeddings, axis=0)

embedding_dim = word2vec_model.vector_size
train['embeddings'] = train['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))
test['embeddings'] = test['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))


# Step 4: Neural Network Architecture
X_train = np.vstack(train['embeddings'])
y_train = train['label']
X_test = np.vstack(test['embeddings'])
y_test = test['label']

val_sentences = [text.split() for text in val['text']]
val_word2vec_model = Word2Vec(val_sentences, min_count=10, sg=1)
val['embeddings'] = val['text'].apply(lambda x: get_sentence_embeddings(x, val_word2vec_model, embedding_dim))
X_val = np.vstack(val['embeddings'])
y_val = val['label']

#get mean, medean, max and min of text in entire dataset i.e. test, train and validation
all_sentences = [text.split() for text in train['text']]
all_sentences.extend([text.split() for text in test['text']])
all_sentences.extend([text.split() for text in val['text']])
#calculate mean, median, max and min of text length
text_lengths = [len(text) for text in all_sentences]
mean_text_length = np.mean(text_lengths)
median_text_length = np.median(text_lengths)
max_text_length = np.max(text_lengths)
min_text_length = np.min(text_lengths)
print("Mean text length: ", mean_text_length)
print("Median text length: ", median_text_length)
print("Max text length: ", max_text_length)
print("Min text length: ", min_text_length)


best_accuracy = 0
best_model = None
best_num_iter = None
for num_iter in range(100, 1000, 100):
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=num_iter, activation='relu', solver='adam', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_num_iter = num_iter

print("Number of Iterations: ",best_model.n_iter_)
#iterate thorugh different embedding sizes, and different number sof iterations
# best_accuracy = 0
# best_model = None
# best_embedding_dim = None
# best_num_iter = None
# for embedding_dim in range(50, 250, 50):
#     sentences = [text.split() for text in train['text']]
#     word2vec_model = Word2Vec(sentences, min_count=10, sg=1)
#     train['embeddings'] = train['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))
#     test['embeddings'] = test['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))
#     val['embeddings'] = val['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))
#     X_train = np.vstack(train['embeddings'])
#     y_train = train['label']
#     X_test = np.vstack(test['embeddings'])
#     y_test = test['label']
#     X_val = np.vstack(val['embeddings'])
#     y_val = val['label']


#     for num_iter in range(100, 1500, 500):
#         model = MLPClassifier(hidden_layer_sizes=(embedding_dim,), max_iter=num_iter, activation='relu', solver='adam', random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_val)
#         accuracy = accuracy_score(y_val, y_pred)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model
#             best_embedding_dim = embedding_dim
#             best_num_iter = num_iter


# model = MLPClassifier(hidden_layer_sizes=(embedding_dim,), max_iter=1000, activation='relu', solver='adam', random_state=42)
# model.fit(X_train, y_train)

# Step 6: Testing and Evaluating
# Test the model
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print("Test Accuracy:", accuracy)
print("Macro F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
