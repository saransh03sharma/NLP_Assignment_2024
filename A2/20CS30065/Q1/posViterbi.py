from utils import *
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# train and test file paths
train_path = './NLP2/NLP2/train.txt'
test_path = './NLP2/NLP2/test.txt'

# print the one sample each from the train and test files
print_train_sample(train_path)
print_train_sample(test_path)


# read the train and test files
train_words_list, train_id_list, train_pos_tags_list = read_txt(train_path)
print("List of Words in first 5 Block:", train_words_list[:5])
print("List of POS Tags in first 5 Block:", train_pos_tags_list[:5])
print("Number of Training Samples: ",len(train_words_list))
print("----------------------------------------------------------------------")


test_words_list, test_id_list, test_pos_tags_list = read_txt(test_path)
print("List of Words in first 5 Block:", test_words_list[:5])
print("List of POS Tags in first 5 Block:", test_pos_tags_list[:5])
print("Number of Test Samples: ",len(test_words_list))
print("----------------------------------------------------------------------")

train_dataset = read_dataset(train_path)
test_dataset = read_dataset(test_path)


# get the vocabulary
vocabulary = get_vocab(train_words_list)
print("Vocabulary (10 entries):", vocabulary[:10])
print("Length of Vocabulary: ",len(vocabulary))
print("----------------------------------------------------------------------")

# get the unique POS tags
unique_tags = get_unique_tags(train_pos_tags_list)
print("Unique POS Tags:", unique_tags)
print("Number of unique POS tags: ",len(unique_tags))
print("----------------------------------------------------------------------")

# get the transition counts
train_tag_transition = get_tag_count_transition(train_words_list, train_pos_tags_list)
print("Number of Transition Counts: ",len(train_tag_transition))
print("----------------------------------------------------------------------")
      
# get the emission counts
train_tag_emission = get_tag_count_emission(train_words_list, train_pos_tags_list)
print("Number of Emission Counts: ",len(train_tag_emission))
print("----------------------------------------------------------------------")
  
# get the tag pair counts  
train_tag_pair_counts = get_tag_pair_counts(train_words_list,train_pos_tags_list)
print("Number of Tag Pair Counts: ",len(train_tag_pair_counts))
print("----------------------------------------------------------------------")


# get the tag word counts
train_tag_word_counts = get_tag_word_pair(train_words_list,train_pos_tags_list)
print("Number of Tag Word Counts: ",len(train_tag_word_counts))
print("----------------------------------------------------------------------")

#initialize the predicted and actual tags lists
predicted_tags = []
actual_tags = []

# get the number of unique tags and vocabulary
T = len(unique_tags)
V = len(vocabulary)

#flush viterbi_predictions_test/train.tsv
open('viterbi_predictions_test.tsv', 'w').close()
open('viterbi_predictions_train.tsv', 'w').close()
    
print("Note: Creation of TRAIN TSV might take some time")
print()
# run the viterbi algorithm on the training data
for i in range(len(train_words_list)):
    sentence = train_words_list[i]
    pos = train_pos_tags_list[i]
    sent_id = train_id_list[i]

    best_tags = viterbi_algorithm(sentence, train_tag_transition, train_tag_pair_counts, train_tag_emission, train_tag_word_counts,T,V,'train',vocabulary)
    assert len(best_tags) == len(pos), "Length of predicted tags and actual tags should be same"
    predicted_tags.extend(best_tags)
    actual_tags.extend(pos)
    create_tsv(sent_id,train_dataset[sent_id], predicted_tags, 'train')

assert len(predicted_tags) == len(actual_tags), "Length of predicted tags and actual tags should be same"

print("Train TSV Created")
print()
# calculate the f1, precision, recall and accuracy scores
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
f1 = f1_score(np.array(actual_tags), np.array(predicted_tags), average='weighted')
precision = precision_score(np.array(actual_tags), np.array(predicted_tags), average='weighted')
recall = recall_score(np.array(actual_tags), np.array(predicted_tags), average='weighted')
accuracy = accuracy_score(np.array(actual_tags), np.array(predicted_tags))

print("Inference Results on Training Data")
print("F1 Score: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)
print("------------------------------------------------------------")

#initialize the predicted and actual tags lists
predicted_tags = []
actual_tags = []

print("Note: Creation of TEST TSV might take some time")
print()
# run the viterbi algorithm on the test data
for i in range(len(test_words_list)):
    sentence = test_words_list[i]
    pos = test_pos_tags_list[i]
    sent_id  = test_id_list[i]
    best_tags = viterbi_algorithm(sentence, train_tag_transition, train_tag_pair_counts, train_tag_emission, train_tag_word_counts,T,V,'test',vocabulary)
    assert len(best_tags) == len(pos), "Length of predicted tags and actual tags should be same"
    predicted_tags.extend(best_tags)
    actual_tags.extend(pos)
    create_tsv(sent_id,test_dataset[sent_id], predicted_tags, 'test')

assert len(predicted_tags) == len(actual_tags), "Length of predicted tags and actual tags should be same"

print("Test TSV Created")
print()
# calculate the f1, precision, recall and accuracy scores
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
f1 = f1_score(np.array(actual_tags), np.array(predicted_tags), average='weighted')
precision = precision_score(np.array(actual_tags), np.array(predicted_tags), average='weighted')
recall = recall_score(np.array(actual_tags), np.array(predicted_tags), average='weighted')
accuracy = accuracy_score(np.array(actual_tags), np.array(predicted_tags))


print("Inference Results on Test Data")
print("F1 Score: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)