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
from nltk.stem import WordNetLemmatizer, PorterStemmer
import warnings
warnings.filterwarnings('ignore')
import joblib


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

#display data
print(train.head())
print("Number of rows in train data: ", train.shape[0])
print(test.head())
print("Number of rows in test data: ", test.shape[0])
print(val.head())
print("Number of rows in validation data: ", val.shape[0])
print('-'*100)
print()

#print column names
print("Column names: ", train.columns)
print('-'*100)
print()

#preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.replace('\\', ' ')

    #remove stopwords
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


# Step: Word2Vec Embeddings
train_sentences = [text.split() for text in train['text']]
val_sentences = [text.split() for text in val['text']]
test_sentences = [text.split() for text in test['text']]

#print mean median min and max length of all sentences
sentence_lengths = [len(sentence) for sentence in train_sentences]
sentence_lengths.extend([len(sentence) for sentence in val_sentences])
sentence_lengths.extend([len(sentence) for sentence in test_sentences])
print("Mean Sentence Length: ", np.mean(sentence_lengths))
print("Median Sentence Length: ", np.median(sentence_lengths))
print("Min Sentence Length: ", np.min(sentence_lengths))
print("Max Sentence Length: ", np.max(sentence_lengths))
print('-'*100)
print()

words = []
for sentence in train_sentences:
    words.extend(sentence)
word_freq = pd.Series(words).value_counts()
words = word_freq[(word_freq >= 10) & (word_freq <= 80)].index.tolist()
vocab = set(words)
print("Vocabulary Size: ", len(vocab))
print('-'*100)
print()

word2vec_model = Word2Vec(train_sentences, min_count=10, sg=1)


# Step: Sentence Embeddings
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
print("Embedding Dimension: ", embedding_dim)
train['embeddings'] = train['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))
test['embeddings'] = test['text'].apply(lambda x: get_sentence_embeddings(x, word2vec_model, embedding_dim))


# Step: Neural Network Architecture
X_train = np.vstack(train['embeddings'])
y_train = train['label']
X_test = np.vstack(test['embeddings'])
y_test = test['label']

val_sentences = [text.split() for text in val['text']]
val_word2vec_model = Word2Vec(val_sentences, min_count=10, sg=1)
val['embeddings'] = val['text'].apply(lambda x: get_sentence_embeddings(x, val_word2vec_model, embedding_dim))
X_val = np.vstack(val['embeddings'])
y_val = val['label']


# Step: Train Neural Network
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
y_pred = best_model.predict(X_test)
print('-'*100)
print()

#save the predictions to csv
test['preds'] = y_pred
test = test.drop(columns=['embeddings'])
test.to_csv('w2v_test.csv')

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
print('-'*100)
print()

#save model
joblib.dump(best_model, 'model_part1_a.pkl')
