import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
from transformers import RobertaModel, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
from torch import cuda
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if cuda.is_available() else 'cpu'


train_path = './NLP3/train.csv'
test_path = './NLP3/test.csv'

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


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\\', ' ')

    lemmatizer = WordNetLemmatizer()
    preprocessed_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return preprocessed_text

train['text'] = train['text'].apply(preprocess_text)
val['text'] = val['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

# Defining some key variables that will be used later on in the training
MAX_LEN = 100
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]['text'])
        label = self.data.iloc[index]['label']
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]



        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label)
        }

training_set = SentimentData(train, tokenizer, MAX_LEN)
testing_set = SentimentData(test, tokenizer, MAX_LEN)
validation_set = SentimentData(val, tokenizer, MAX_LEN)

print("TRAIN Dataset: {}".format(train.shape))
print("TEST Dataset: {}".format(test.shape))
print("VAL Dataset: {}".format(val.shape))

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }


val_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **val_params)


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


def train_bert(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    epoch_losses = []
    epoch_accuracies = []
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return epoch_loss, epoch_accu


best_accuracy = 0
best_epoch = 0
best_model = None

for num_epoch in range(10, 50, 5):
    losses = []
    accuracies = []
    loss, accuracy = train_bert(num_epoch)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = num_epoch
        best_model = model
    losses.append(loss)
    accuracies.append(accuracy)


model = best_model
plt.plot(range(1, best_epoch + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.legend()
plt.show()


def evaluate(model, testing_loader, type, save_csv=False):
    model.eval()
    n_correct = 0
    preds = []
    true_labels = []
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            preds.extend(big_idx.cpu().detach().numpy())
            true_labels.extend(targets.cpu().detach().numpy())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                if type == "valid":
                    print(f"Validation Loss per 100 steps: {loss_step}")
                    print(f"Validation Accuracy per 100 steps: {accu_step}")
                else:
                    print(f"Testing Loss per 100 steps: {loss_step}")
                    print(f"Testing Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    if type == "valid":
        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_accu}")
    else:
        print(f"Testing Loss Epoch: {epoch_loss}")
        print(f"Testing Accuracy Epoch: {epoch_accu}")


    return epoch_accu, preds, true_labels

# Evaluate the model
accuracy, preds, true_labels = evaluate(model, testing_loader, type="test", save_csv=True)

# Print accuracy
print("Testing Accuracy:", accuracy)

# Calculate and print macro F1 score
macro_f1 = f1_score(true_labels, preds, average='macro')
print("Macro F1 Score:", macro_f1)

# Print confusion matrix
conf_matrix = confusion_matrix(true_labels, preds)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(true_labels, preds)
print("Classification Report:")
print(class_report)

torch.save(model.state_dict(), './bert.pt')

test['preds'] = preds

test.head()

test.to_csv('./bert.csv')

