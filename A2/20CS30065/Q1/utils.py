import math
# Function to print the first  sentences of the training data
def print_train_sample(path):
    with open(path, 'r') as file:
        inside_block = False
        for line in file:
            if line.startswith("# sent_id"):
                if inside_block:
                    break
                print(line.strip())  # Print the sent_id line
                inside_block = True  # Set inside_block to True
            elif inside_block and line.strip() and not line.startswith("#"):
                print(line.strip())
    print("----------------------------------------------------------------------")

# function to read the txt file
def read_txt(path):
    words_list = []
    pos_tags_list = []
    id_list = []

    with open(path, 'r', encoding='cp1252', errors='ignore') as file:
      current_block_words = []
      current_block_pos_tags = []
      current_id = None

      for line in file:
          if line.startswith("# sent_id"):
              if current_block_words:
                  words_list.append(current_block_words)
                  pos_tags_list.append(current_block_pos_tags)
                  current_block_words = []
                  current_block_pos_tags = []
                  id_list.append(current_id)
              current_id = line.strip().split("=")[-1].strip()

          elif line.strip() and not line.startswith("#"):
              tokens = line.strip().split()
              if tokens[0].isdigit():
                    word = tokens[1]
                    word_id = tokens[0]
                    pos_tag = tokens[3]
                    current_block_words.append(word)
                    current_block_pos_tags.append(pos_tag)

    if current_block_words:
          words_list.append(current_block_words)
          pos_tags_list.append(current_block_pos_tags)
          id_list.append(current_id)
    return words_list, id_list, pos_tags_list

# function to get the vocabulary
def get_vocab(words_list):
    vocabulary = set()
    for block_words in words_list:
        vocabulary.update(block_words)
    vocabulary = list(vocabulary)
    return vocabulary


# define func to add <s> and </s> to each block
def add_start_end_tags(pos_tags_list):
    for i in range(len(pos_tags_list)):
        pos_tags_list[i] = ['<s>'] + pos_tags_list[i] + ['</s>']
    return pos_tags_list

# function to get the tag counts to be used while computing transition probabilites
def get_tag_count_transition(word_list, pos_tags_list):
    transition_tag_count = {}
    for j in range(len(word_list)):
        sentence = word_list[j]
        pos_tag = pos_tags_list[j]
        for i in range(len(sentence)):
            if i == 0:
                if "<s>" not in transition_tag_count:
                    transition_tag_count["<s>"] = 0
                transition_tag_count["<s>"] += 1  
            else:
                if pos_tag[i-1] not in transition_tag_count:
                    transition_tag_count[pos_tag[i-1]] = 0
                transition_tag_count[pos_tag[i-1]] += 1
                
        if pos_tag[-1] not in transition_tag_count:
            transition_tag_count[pos_tag[-1]] = 0
        transition_tag_count[pos_tag[-1]] += 1
    return transition_tag_count

# function to get tag counts to be used to compute emission probabilities
def get_tag_count_emission(word_list, pos_tags_list):
    emission_tag_count = {}
    for j in range(len(word_list)):
        sentence = word_list[j]
        pos_tag = pos_tags_list[j]
        for i in range(len(sentence)):
            if pos_tag[i] not in emission_tag_count:
                emission_tag_count[pos_tag[i]] = 0
            emission_tag_count[pos_tag[i]] += 1

    return emission_tag_count

# function to get the tag pair counts to be used while computing transition probabilities
def get_tag_pair_counts(word_list, pos_tags_list):
    transition_tag_pair_count = {}
    for j in range(len(word_list)):
        sentence = word_list[j]
        pos_tag = pos_tags_list[j]
        for i in range(len(sentence)):
            if i == 0:
                if "<s>" not in transition_tag_pair_count:
                    transition_tag_pair_count["<s>"] = {}
                if pos_tag[i] not in transition_tag_pair_count['<s>']:
                    transition_tag_pair_count["<s>"][pos_tag[i]] = 0
                transition_tag_pair_count["<s>"][pos_tag[i]] += 1  
            else:
                if pos_tag[i-1] not in transition_tag_pair_count:
                    transition_tag_pair_count[pos_tag[i-1]] = {}
                if pos_tag[i] not in transition_tag_pair_count[pos_tag[i-1]]:
                    transition_tag_pair_count[pos_tag[i-1]][pos_tag[i]] = 0
                transition_tag_pair_count[pos_tag[i-1]][pos_tag[i]] += 1
                
        if pos_tag[-1] not in transition_tag_pair_count:
            transition_tag_pair_count[pos_tag[-1]] = {}
        if "</s>" not in transition_tag_pair_count[pos_tag[-1]]:
            transition_tag_pair_count[pos_tag[-1]]['</s>'] = 0
        transition_tag_pair_count[pos_tag[-1]]['</s>'] += 1
    return transition_tag_pair_count


# function to get the tag word pair counts to be used while computing emission probabilities
def get_tag_word_pair(word_list, pos_tags_list):
    emission_tag_pair_count = {}
    for j in range(len(word_list)):
        sentence = word_list[j]
        pos_tag = pos_tags_list[j]
        for i in range(len(sentence)):
            if pos_tag[i] not in emission_tag_pair_count:
                emission_tag_pair_count[pos_tag[i]] = {}
            if sentence[i] not in emission_tag_pair_count[pos_tag[i]]:
                emission_tag_pair_count[pos_tag[i]][sentence[i]] = 0
            emission_tag_pair_count[pos_tag[i]][sentence[i]] += 1
            
    return emission_tag_pair_count

# function to get the unique tags
def get_unique_tags(pos_tags_list):
    unique_tags = set()
    unique_tags.add('<s>')
    unique_tags.add('</s>')
    for block_tags in pos_tags_list:
        unique_tags.update(block_tags)
    unique_tags = list(unique_tags)
    return unique_tags

# function to run the viterbi algorithm
def viterbi_algorithm(sentence, tag_transition, tag_pair_counts, tag_emission, tag_word_counts,T,V, mode,vocabulary):
    
    pos_tags = [None]*(len(sentence))
    backpointer = [{} for i in range(len(sentence) + 1)]
    
    scores = []
    prev_layer_prob = {}
    prev_layer_prob['<s>'] = 1
    
    for i in range(len(sentence)):
        curr_layer_prob = {}
        for prev_tag in prev_layer_prob.keys():
            if prev_tag in tag_pair_counts:
                for next_tag in tag_pair_counts[prev_tag].keys():
                    
                    if next_tag not in curr_layer_prob:
                        curr_layer_prob[next_tag] = -math.inf
                    
                    
                    if mode == 'train':
                        if prev_tag not in tag_transition or next_tag not in tag_pair_counts[prev_tag] or next_tag not in tag_word_counts or sentence[i] not in tag_word_counts[next_tag]:
                            prob_score = -math.inf
                        else:
                            prob_score = prev_layer_prob[prev_tag] + math.log(tag_pair_counts[prev_tag][next_tag]) - math.log(tag_transition[prev_tag])  + math.log(tag_word_counts[next_tag][sentence[i]]) - math.log(tag_emission[next_tag])
                    else:
                        if prev_tag not in tag_transition or next_tag not in tag_pair_counts[prev_tag] or next_tag not in tag_word_counts:
                            prob_score = -math.inf
                        
                        elif sentence[i] not in vocabulary or sentence[i] not in tag_word_counts[next_tag]:
                            prob_score = prev_layer_prob[prev_tag] + math.log(tag_pair_counts[prev_tag][next_tag]) - math.log(tag_transition[prev_tag]) + math.log(1) - math.log(tag_emission[next_tag] + V)
                        
                        else:
                            prob_score = prev_layer_prob[prev_tag] + math.log(tag_pair_counts[prev_tag][next_tag]) - math.log(tag_transition[prev_tag])  + math.log(tag_word_counts[next_tag][sentence[i]] + 1) - math.log(tag_emission[next_tag] + V)
                    
                    if prob_score > curr_layer_prob[next_tag]:
                        curr_layer_prob[next_tag] = prob_score
                        backpointer[i][next_tag] = prev_tag

        prev_layer_prob = curr_layer_prob
        scores.append(prev_layer_prob)
    
    # End State
    i = len(sentence)
    curr_layer_prob = {}
    for prev_tag in prev_layer_prob.keys():
        if prev_tag in tag_pair_counts:
            next_tag = "</s>"
            if next_tag not in curr_layer_prob:
                curr_layer_prob[next_tag] = -math.inf

            if prev_tag not in tag_pair_counts or next_tag not in tag_pair_counts[prev_tag]:
                prob_score = -math.inf
            else:
                prob_score = prev_layer_prob[prev_tag] + math.log(tag_pair_counts[prev_tag][next_tag]) - math.log(tag_transition[prev_tag])
            
            if prob_score > curr_layer_prob[next_tag]:
                curr_layer_prob[next_tag] = prob_score
                backpointer[i][next_tag] = prev_tag
    prev_layer_prob = curr_layer_prob
    scores.append(prev_layer_prob)
    # Backtrack
    max_tag = backpointer[-1]["</s>"]
    pos_tags[-1] = max_tag
    for i in range(len(sentence)-1, 0, -1):
        pos_tags[i-1] = backpointer[i][pos_tags[i]]
    return pos_tags

def read_dataset(file_path):
    data = {}
    with open(file_path, 'r', encoding='cp1252', errors='ignore') as file:
        current_sentence = {}
        for line in file:
            line = line.strip()
            if line.startswith('# sent_id'):
                if current_sentence:
                    data[current_sentence['sent_id']] = current_sentence
                    current_sentence = {}
                current_sentence['sent_id'] = line.split('=')[1].strip()
            elif line.startswith('# text'):
                current_sentence['text'] = line.split('=')[1].strip()
            elif not line.startswith('#'):
                if line.strip() != '':
                    parts = line.split()
                    if parts[0].isdigit() and parts[4].isdigit():  
                        token_id = int(parts[0])
                        word = parts[1]
                        normalized_word = parts[2]
                        pos_tag = parts[3]
                        head = int(parts[4]) if parts[4] != '0' else None
                        dep_relation = parts[5]
                        #if head or dep_relation doesnt exists skip this sentence
                        if dep_relation == '':
                            continue
                        current_sentence[token_id] = {'word': word,'normalized':normalized_word, 'pos_tag': pos_tag, 'head': head, 'dep_relation': dep_relation, 'index':token_id}
        if current_sentence:
            data[current_sentence['sent_id']] = current_sentence
    return data


# function to create the tsv file
def create_tsv(sent_id, sample, predicted_tags,type):
    if type == 'train':
        with open('viterbi_predictions_train.tsv', 'a') as file:
            for i in range(1,len(sample)-1):
                file.write(sent_id + '\t' + str(i) + '\t' + sample[i]['word'] +  '\t' + predicted_tags[i-1] + '\n')
    else:
        with open('viterbi_predictions_test.tsv', 'a') as file:
            for i in range(1,len(sample)-1):
                file.write(sent_id + '\t' + str(i) + '\t' + sample[i]['word'] +  '\t' + predicted_tags[i-1] + '\n')