import numpy as np
LEFT_ARC = 0
RIGHT_ARC = 1
REDUCE = 2
SHIFT = 3


def read_dataset(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
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

def get_pos_tags(dataset):
    pos_tags = set()
    for id in dataset:
        sentence = dataset[id]
        for token in sentence.values():
            if type(token) == dict:
                pos_tags.add(token['pos_tag'])
    return sorted(list(pos_tags))

def get_dep_relations(dataset):
    dep_relations = set()
    for id in dataset:
        sentence = dataset[id]
        for token in sentence.values():
            if type(token) == dict:
                dep_relations.add(token['dep_relation'])
    return sorted(list(dep_relations))

def get_most_frequent_words(dataset):
    word_count = {}
    for id in dataset:
        sentence = dataset[id]
        for token in sentence.values():
            if type(token) == dict:
                word = token['normalized']
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
    most_frequent_words = [k for k, v in word_count.items() if v <= len(dataset) / 2][:1000]
    return sorted(most_frequent_words)

def one_hot_encoding_pos_tag(tag, pos_tags):
    if tag not in pos_tags:
        return np.zeros(len(pos_tags))
    one_hot = np.zeros(len(pos_tags))
    one_hot[pos_tags.index(tag)] = 1
    return one_hot

def one_hot_encoding_dep_relation(rel, dep_relations):
    if rel not in dep_relations:
        return np.zeros(len(dep_relations))
    one_hot = np.zeros(len(dep_relations))
    one_hot[dep_relations.index(rel)] = 1
    return one_hot

def one_hot_encoding_word(word, most_frequent_words):
    if word not in most_frequent_words:
        return np.zeros(len(most_frequent_words))
    one_hot = np.zeros(len(most_frequent_words))
    one_hot[most_frequent_words.index(word)] = 1
    return one_hot

def oracle(config,word_head,word_index):
    buffer, stack, relations = config
    if len(stack) > 0 and len(buffer) > 0:
        top_stack = stack[-1]
        first_buffer = buffer[0]
        flag = False
        
        #see if head of top of stack is discovered
        for relation in relations:
            if top_stack == relation[2]:
                flag = True
        
        if (word_head[top_stack] == word_index[first_buffer]) and (flag == False):
            return LEFT_ARC
        
        elif (word_head[first_buffer] == word_index[top_stack]):
            return RIGHT_ARC
        
        elif (flag == True):
        #if head of top of stack is discovered and a relation exists involving first buffer
            for relation in relations:
                if (first_buffer == relation[0] and relation[2] != top_stack) or (first_buffer == relation[2] and relation[0] != top_stack):
                    return REDUCE
            return SHIFT
        
        else:
            return SHIFT
    
    elif len(buffer) > 0:
        return SHIFT

    else:
        return SHIFT

def modified_configuration(config,transistion, word_dep):
    buffer, stack, relations = config
    
    if transistion == LEFT_ARC:
        if len(stack) > 0 and len(buffer) > 0:
            top_stack = stack[-1]
            first_buffer = buffer[0]
            relations.append((first_buffer,word_dep[top_stack],top_stack))
            stack.pop()
    
    elif transistion == RIGHT_ARC:
        if len(stack) > 0 and len(buffer) > 0:
            top_stack = stack[-1]
            first_buffer = buffer[0]
            relations.append((top_stack,word_dep[first_buffer],first_buffer))
            stack.append(buffer.pop(0))
    
    elif transistion == REDUCE:
        if len(stack) > 0:
            stack.pop()
    
    elif transistion == SHIFT:
        if len(buffer) > 0:
            stack.append(buffer.pop(0))
    else:
        raise ValueError(f"Invalid transistion: {transistion}")
    return (buffer, stack, relations)

def get_word_pos(sentence):
    word_pos = {}
    for token in sentence.values():
        if type(token) == dict:
            word_pos[token['normalized']] = token['pos_tag']
    return word_pos

def get_word_head(sentence):
    word_head = {}
    for token in sentence.values():
        if type(token) == dict:
            word_head[token['normalized']] = token['head']
    return word_head

def get_word_dep(sentence):
    word_dep = {}
    for token in sentence.values():
        if type(token) == dict:
            word_dep[token['normalized']] = token['dep_relation']
    return word_dep

def get_word_index(sentence):
    word_index = {}
    for token in sentence.values():
        if type(token) == dict:
            word_index[token['normalized']] = token['index']
    return word_index

def create_dataset(dataset):
    training_sample = []
    for i in dataset:
        sentence = dataset[i]
        id = dataset[i]['sent_id']
        buffer = [token['normalized'] for token in sentence.values() if type(token) == dict]
        stack = []
        relations = []
        
        word_head = get_word_head(sentence)
        word_index =get_word_index(sentence)
        word_dep = get_word_dep(sentence)
                
        config = (buffer, stack, relations)
        while len(buffer):    
            ground_transistion = oracle(config,word_head,word_index)
            training_sample.append((config,ground_transistion,id))
            config = modified_configuration(config,ground_transistion,word_dep)
            buffer, stack, relations = config
    return training_sample


def get_features(config, word_pos,word_index, pos_tag, word_head, dep_relation, most_frequent_words,P,V,R):
    buffer, stack, relations = config
    if len(stack) > 0:
        top_stack = stack[-1]
        
        top_stack_pos = one_hot_encoding_pos_tag(word_pos[top_stack], pos_tag)
        top_stack_word = one_hot_encoding_word(top_stack, most_frequent_words)
        top_stack_dep = np.zeros(R)
        top_stack_left_dep = np.zeros(R)
        top_stack_right_dep = np.zeros(R)
        
        for relation in relations:
            if (relation[2] == top_stack and word_head[top_stack] == relation[0]):
                top_stack_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
            
        left = 1000
        right = 0

        for relation in relations:
            if relation[0] == top_stack and word_index[relation[2]] < word_index[top_stack]:
                if top_stack_left_dep.sum() == 0:
                    top_stack_left_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
                    left = word_index[relation[2]]
                else:
                    if word_index[relation[2]] < left:
                        top_stack_left_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
                        left = word_index[relation[2]]
            
            elif relation[0] == top_stack and word_index[relation[2]] > word_index[top_stack]:
                if top_stack_right_dep.sum() == 0:
                    top_stack_right_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
                    right = word_index[relation[2]]
                else:
                    if word_index[relation[2]] > right:
                        top_stack_right_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
                        right = word_index[relation[2]]
    else:
        top_stack_pos = np.zeros(P)
        top_stack_word = np.zeros(V)
        top_stack_left_dep = np.zeros(R)
        top_stack_right_dep = np.zeros(R)
        top_stack_dep = np.zeros(R)
    
    if len(buffer) > 0:
        first_buffer = buffer[0]
        first_buffer_pos = one_hot_encoding_pos_tag(word_pos[first_buffer], pos_tag)
        first_buffer_word = one_hot_encoding_word(first_buffer, most_frequent_words)
        first_buffer_left_dep = np.zeros(R)

        left = 1000
        for relation in relations:
            if relation[0] == first_buffer and word_index[relation[2]] < word_index[first_buffer]:
                if first_buffer_left_dep.sum() == 0:
                    first_buffer_left_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
                    left = word_index[relation[2]]
                else:
                    if word_index[relation[2]] < left:
                        first_buffer_left_dep = one_hot_encoding_dep_relation(relation[1], dep_relation)
                        left = word_index[relation[2]]

    else:
        first_buffer_pos = np.zeros(P)
        first_buffer_word = np.zeros(V)
        first_buffer_left_dep = np.zeros(R)

    if len(buffer) > 1:
        look_buffer_pos = one_hot_encoding_pos_tag(word_pos[buffer[1]], pos_tag)
    else:
        look_buffer_pos = np.zeros(P)
    
    assert len(top_stack_pos) == P
    assert len(first_buffer_pos) == P
    assert len(look_buffer_pos) == P
    assert len(top_stack_word) == V
    assert len(first_buffer_word) == V
    assert len(top_stack_left_dep) == R
    assert len(top_stack_right_dep) == R
    assert len(first_buffer_left_dep) == R
    assert len(top_stack_dep) == R
    
    feat = np.concatenate([top_stack_word, top_stack_pos, top_stack_dep, top_stack_left_dep, top_stack_right_dep, first_buffer_word, first_buffer_pos, first_buffer_left_dep, look_buffer_pos])
    assert len(feat) == (2 * V + 3 * P + 4 * R)
    return feat

def get_best_transition(features, weight):
    scores = np.dot(features, weight)
    return np.argmax(scores)

def update_weights(weight, features, ground_transistion, predicted_transistion):
    weight = weight + features[ground_transistion] - features[predicted_transistion]
    return weight

def get_most_common_dependency_relation(train_dataset):
    pos_pos_dep = {}
    for id in train_dataset:
        sentence = train_dataset[id]
        for i in range(1,len(sentence)-1):
            word = sentence[i]
            if word['head'] == None:
                continue
            head = sentence[int(word['head'])]
            pos_x = word['pos_tag']
            pos_y = head['pos_tag']
            dep_relation = word['dep_relation']
            key = (pos_x,pos_y)
            if key not in pos_pos_dep:
                pos_pos_dep[key] = {}
            if dep_relation not in pos_pos_dep[key]:
                pos_pos_dep[key][dep_relation] = 0
            pos_pos_dep[key][dep_relation] += 1
    for key in pos_pos_dep:
        dep_relation = max(pos_pos_dep[key], key=pos_pos_dep[key].get)
        pos_pos_dep[key] = dep_relation
    return pos_pos_dep

def arc_eager(train_dataset, config_transistion,pos_tag,dep_relation,most_frequent_words,P,V,R):
    weight = np.full((2 * V + 3 * P + 4 * R),-1e3,dtype=float)   
    
    for i in range(len(config_transistion)):
        config = config_transistion[i][0]
        ground_transistion = config_transistion[i][1]

        id = config_transistion[i][2]
        sentence = train_dataset[id]
        
        word_pos = get_word_pos(sentence)
        word_head = get_word_head(sentence)
        word_dep = get_word_dep(sentence)
        word_index = get_word_index(sentence)

        features = []
        for i in range(4):
            transition = i
            updated_config = modified_configuration(config,transition,word_dep)
            features.append(get_features(updated_config, word_pos,word_index, pos_tag, word_head, dep_relation, most_frequent_words,P,V,R))
        
        predicted_transistion = get_best_transition(features, weight)
        weight = update_weights(weight, features, ground_transistion, predicted_transistion)

    return weight

def modified_configuration_test(config,transistion,tag_dep_relation,word_pos):
    buffer, stack, relations = config
    if transistion == LEFT_ARC:
        if len(stack) > 0 and len(buffer) > 0:
            top_stack = stack[-1]
            first_buffer = buffer[0]
            if (word_pos[top_stack],word_pos[first_buffer]) in tag_dep_relation:
                dep_relation = tag_dep_relation[(word_pos[top_stack],word_pos[first_buffer])]
            else:
                dep_relation = 'None'
            relations.append((first_buffer,dep_relation,top_stack))
            stack.pop()
    elif transistion == RIGHT_ARC:
        if len(stack) > 0 and len(buffer) > 0:
            top_stack = stack[-1]
            first_buffer = buffer[0]
            if (word_pos[first_buffer],word_pos[top_stack]) in tag_dep_relation:
                dep_relation = tag_dep_relation[(word_pos[first_buffer],word_pos[top_stack])]
            else:
                dep_relation = 'None'
            relations.append((top_stack,dep_relation,first_buffer))
            stack.append(buffer.pop(0))
    elif transistion == REDUCE:
        if len(stack) > 0:
            stack.pop()
    elif transistion == SHIFT:
        if len(buffer) > 0:
            stack.append(buffer.pop(0))
    else:
        raise ValueError(f"Invalid transistion: {transistion}")
    return (buffer, stack, relations)

def get_index_word(sentence):
    word_index = {}
    for token in sentence.values():
        if type(token) == dict:
            word_index[token['index']] = token['normalized']
    return word_index

def test_model(test_dataset,pos_tag,dep_relation,tag_dep_relation,most_frequent_words,P,V,R, weight):
    val = []
    for id in test_dataset:
        sentence = test_dataset[id]
        word_pos = get_word_pos(sentence)
        index_word = get_index_word(sentence)
        word_head = get_word_head(sentence)
        word_index = get_word_index(sentence)
        buffer = [token['normalized'] for token in sentence.values() if type(token) == dict]
        stack = []
        relations = []
        
        while(buffer):
            config = (buffer, stack, relations)

            features = []
            for i in range(4):
                transition = i
                updated_config = modified_configuration_test(config,transition,tag_dep_relation,word_pos)
                features.append(get_features(updated_config, word_pos,word_index, pos_tag, word_head, dep_relation, most_frequent_words,P,V,R))    
            
            predicted_transistion = get_best_transition(features, weight)
            
            top_stack = stack[-1] if len(stack) > 0 else None
            if predicted_transistion == RIGHT_ARC:
                if len(stack)==0:
                    predicted_transistion = SHIFT
            
            if predicted_transistion == LEFT_ARC:
                if len(stack)==0:
                    predicted_transistion = SHIFT
                for relation in relations:
                    if relation[2] == top_stack:
                        predicted_transistion = REDUCE
                        break
                
            if predicted_transistion == REDUCE:
                if len(stack)==0:
                    predicted_transistion = SHIFT
                flag = 0
                for relation in relations:
                    if relation[2] == top_stack:
                        flag = 1
                        break
                if flag == 0:
                    predicted_transistion = LEFT_ARC

            updated_config  = modified_configuration_test(config,predicted_transistion,tag_dep_relation,word_pos)
            buffer, stack, relations = updated_config
        
        word_predicted_head = {}
        for relation in relations:
           word_predicted_head[relation[2]] = word_index[relation[0]]   
        

        correct_heads = 0
        total_tokens = 0

        # Calculate UAS for the current sentence
        for token_id, head_id in word_predicted_head.items():
            total_tokens += 1
            if word_head[token_id] == head_id:
                correct_heads += 1
        val.append(correct_heads / total_tokens if total_tokens > 0 else 0)

        #all words in sentence that have no predicted head are assigned head as 0
        for token_id in word_index.values():
            if index_word[token_id] not in word_predicted_head:
                word_predicted_head[index_word[token_id]] = 0

        #create a file dependency_predictions_on.tsv with format sent_id \t token_id \t predicted_head
        with open('dependency_predictions_on.tsv', 'a', encoding="utf-8") as file:
            for token_id, head_id in word_predicted_head.items():
                file.write(f"{id}\t{token_id}\t{head_id}\n")

    # Calculate UAS averaged over all sentences
    uas = np.mean(val)
    return uas