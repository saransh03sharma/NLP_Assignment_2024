from utils import *
import math
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
np.random.seed(42)

train_path = './NLP2/NLP2/train.txt'
test_path = './NLP2/NLP2/test.txt'

LEFT_ARC = 0
RIGHT_ARC = 1
REDUCE = 2
SHIFT = 3

train_dataset = read_dataset(train_path)
print("Number of sentences in train dataset: ", len(train_dataset))
print()
print("-" * 100)
print()

test_dataset = read_dataset(test_path)
print("Number of sentences in test dataset: ", len(test_dataset))
print()
print("-" * 100)
print()

pos_tags = get_pos_tags(train_dataset)
P = len(pos_tags)
print("First 10 POS tags: ", pos_tags[:10])
print("Number of POS tags (P): ", len(pos_tags))
print()
print("-" * 100)
print()

dep_relations = get_dep_relations(train_dataset)
R = len(dep_relations)
print("First 10 dependency relations: ", dep_relations[:10])
print("Number of dependency relations (R): ", len(dep_relations))
print()
print("-" * 100)
print()


most_frequent_words = get_most_frequent_words(train_dataset)
V = len(most_frequent_words)
print("First 10 most frequent words: ", most_frequent_words[:10])
print("Number of most frequent words: ", len(most_frequent_words))
print()
print("-" * 100)
print()


training_config_transistion = create_dataset(train_dataset)
print("Number of sentences in training configuration dataset: ", len(training_config_transistion))
print("-" * 100)
weight = arc_eager(train_dataset,training_config_transistion,pos_tags,dep_relations,most_frequent_words,P,V,R)
print("Dimension of weight vector: ", len(weight))

#save the weight vector in dependency_model_on.npy
np.save('dependency_model_on.npy', weight)

#flush the content in dependency_predictions_on.tsv
open('dependency_predictions_on.tsv', 'w').close()

tag_dep_relation = get_most_common_dependency_relation(train_dataset)
uas = test_model(test_dataset,pos_tags,dep_relations,tag_dep_relation,most_frequent_words,P,V,R, weight)
print("Unlabeled attachment score (UAS): ", uas)
print("-" * 100)
print()