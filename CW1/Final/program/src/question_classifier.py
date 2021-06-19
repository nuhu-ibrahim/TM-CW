import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import codecs
import yaml
import re
import os
import argparse
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# fix the random seed
torch.manual_seed(1)
random.seed(1)

# check if GPU is available
use_gpu = torch.cuda.is_available()

# read .yaml file
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as yml_file:
        config_lines = yaml.load_all(yml_file, Loader=yaml.SafeLoader)
        config_inf = config_lines.__next__()

    return config_inf


# get parameters from the command
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='Training mode - model is saved')
parser.add_argument('--test', action='store_true',
                    help='Testing mode - needs a model to load')
parser.add_argument('--config', type=str, required=True,
                    help='Training mode - Configuration file')
args = parser.parse_args()

# read config information
path = args.config
__file__ = os.getcwd()
current_path = os.path.abspath(__file__)
config_path = current_path + '/../data/' + path
try:
    config = read_yaml(config_path)
except:
    raise SystemExit('Error: Read config file failed! No such a file or file structure wrong.')
try:
    config['preprocess']['lowercase']
    config['preprocess']['remove_stop_words']
    config['preprocess']['remove_special_characters']
    config['preprocess']['replace_abbreviations']
    config['preprocess']['remove_white_space']
except:
    raise SystemExit('Error: Config file wrong! Missing preprocess information.')

if config['preprocess']['lowercase'] not in [True,False]:
    raise SystemExit('Error: lowercase not in [True,False]! Please check the config file.')
if config['preprocess']['remove_stop_words'] not in [True,False]:
    raise SystemExit('Error: lowercase not in [True,False]! Please check the config file.')
if config['preprocess']['remove_special_characters'] not in [True,False]:
    raise SystemExit('Error: lowercase not in [True,False]! Please check the config file.')
if config['preprocess']['replace_abbreviations'] not in [True,False]:
    raise SystemExit('Error: lowercase not in [True,False]! Please check the config file.')
if config['preprocess']['remove_white_space'] not in [True,False]:
    raise SystemExit('Error: lowercase not in [True,False]! Please check the config file.')

# Load the dataset file
def load_file(file_path):
    questions_text = ''
    questions_labels = list()

    questions_file = codecs.open(file_path, 'r', encoding='utf-8')
    question_lines = questions_file.readlines()

    for line in question_lines:
        # clear data
        line = line.strip('\n')
        sentence = clean_data(line.split(' ', 1)[1])

        label = line.split(' ', 1)[0]

        questions_labels.append((sentence.split(" "), label))
        questions_text = questions_text + ' ' + sentence

    questions_file.close()

    # questions_text: string
    # questions_labels: list
    return questions_text, questions_labels


# Load the pre-trained embedding
def load_pretrained_embed(file_path):
    # read pretrained model file
    try:
        glove = codecs.open(file_path, 'r', encoding='utf-8')
    except:
        raise SystemExit('Error: Load pretrained embedding file failed! Please check the config file.')
    word_embed = {}
    glove_lines = glove.readlines()
    # split the pretrained model data
    for line in glove_lines:
        line_list = line.split()
        word = line_list[0]
        embed = line_list[1:]
        embed = [float(num) for num in embed]
        word_embed[word] = embed

    # word_embed: dict
    return word_embed


# Load the stopwords file
def load_stopwords(file_path):
    stopwords = list()
    stopwords_file = codecs.open(file_path, 'r', encoding='utf-8')
    stopwords_lines = stopwords_file.readlines()

    for line in stopwords_lines:
        words = line.split('\n', 1)[0]
        stopwords.append(words)
    stopwords_file.close()

    # stopwords: list
    return stopwords


# Data cleaning function
def clean_data(text):

    # text to lower
    if config['preprocess']['lowercase']:
        text = text_to_lower(text)

    # removing stop words in text
    if config['preprocess']['remove_stop_words']:
        text = ' '.join(remove_stop_word(text.split(' ')))

    # removing special characters
    if config['preprocess']['remove_special_characters']:
        text = remove_special_characters(text)

    # replacing abreviations with full forms
    if config['preprocess']['replace_abbreviations']:
        text = replace_abbreviations_with_full_forms(text)

    # removing all forms of whitespaces
    if config['preprocess']['remove_white_space']:
        text = remove_white_spaces(text)

    # text: string
    return text


# function that removes stop words
def remove_stop_word(words):
    try:
        stopwords = load_stopwords(
            current_path + '/../data/' + config['file_path']['stopwords_path'])
    except:
        raise SystemExit('Error: Load stopwords file failed! Please check the config file.')
    for word in words:
        if word in stopwords:
            words.remove(word)

    # words: list
    return words


# function that replace abbreviations with their full forms
def replace_abbreviations_with_full_forms(text):
    some_abbreviations = ['it\'s', 'how\'s', 'don\'t',
                         'isn\'t', 'he\'s', 'we\'re', 'what\'s', 'who\'s']
    full_forms = ['it is', 'how is', 'do not', 'is not',
                  'he is', 'we are', 'what is', 'who is']

    for i in range(len(some_abbreviations)):
        text = re.sub(some_abbreviations[i], full_forms[i], text)

    # text: string
    return text


# function for removing various forms of whitespaces
def remove_white_spaces(text):
    text = re.sub('(\n|\r|\t|\n\r)', ' ', text)
    text = re.sub('\s+', ' ', text)

    # text: string
    return text.rstrip()


# function for removing special characters from text
def remove_special_characters(text):
    text = re.sub("[,.?!']", "", text)

    # text: string
    return text


# function for changing text to lowercase
def text_to_lower(text):
    text = text.lower()

    # text: string
    return text


# generate word-to-index dictionary
def generate_word_index(text):
    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    vocab = set(text.split())
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    # Adding an embedding for unknown words
    word_to_idx['#UNK#'] = len(word_to_idx)

    # word_to_idx: list of random word embeddings
    return word_to_idx


# convert the sentence to bow vector
def make_bow_vector(sentence, vocab):
    vec = torch.zeros(len(vocab))
    for word in sentence:
        if word in vocab:
            vec[vocab[word]] += 1
        else:
            vec[vocab["#UNK#"]] += 1
    return vec


# convert labels to number list
def make_target(label, label_to_idx):
    return torch.LongTensor([label_to_idx[label]])


# BiLSTM: convert sentences to number list
def prepare_sequence(seq, word_to_idx):
    idxs = []
    for w in seq:
        if config['preprocess']['lowercase']:
            if w.lower() in word_to_idx.keys():
                idxs.append(word_to_idx[w])
            else:
                idxs.append(word_to_idx['#UNK#'])
        else:
            if w in word_to_idx.keys():
                idxs.append(word_to_idx[w])
            else:
                idxs.append(word_to_idx['#UNK#'])
    return torch.tensor(idxs, dtype=torch.long)


# convert tags to id tags
def prepare_tags(seq, tag_to_idx):
    idxs = []
    idxs.append(tag_to_idx[seq])
    return torch.tensor(idxs, dtype=torch.long)


# convert target to idx
def make_target_reverse(index, label_to_idx):
    for i in label_to_idx:
        if label_to_idx[i] == index:
            return i


# Convert the vocabulary into pre-trained embedding
def get_pretrained_vec(word_to_idx, word_embed):

    ix_to_word = {ix: w for w, ix in word_to_idx.items()}
    id_to_embed = {}
    for ix in range(len(word_to_idx)):
        if ix_to_word[ix] in word_embed:
            id_to_embed[ix] = word_embed[ix_to_word[ix]]
        else:
            id_to_embed[ix] = word_embed['#UNK#']

    data = [id_to_embed[ix] for ix in range(len(word_to_idx))]

    # data: list
    return data


# BOW classifier
class BOWModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embed, freeze):
        super(BOWModeler, self).__init__()
        # if don't use pretrained embeddings, randomly initialize the word embeddings
        if pretrained_embed is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # or use pretrained embeddings
        else:
            self.embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embed), freeze=freeze)
        self.fc = nn.Linear(embedding_dim * vocab_size, num_classes)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.fc(embeds)
        log_probs = fun.log_softmax(out, dim=1)
        return log_probs


# LSTM Classifier
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embed, freeze):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # randomly initialize the word embeddings
        if pretrained_embed is None:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # or use pretrained embeddings
        else:
            weight = torch.FloatTensor(pretrained_embed)
            self.word_embeddings = nn.Embedding.from_pretrained(
                weight, freeze=freeze)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, text_lengths, batch_first=True)
        lstm_out, (hidden, cell) = self.lstm(embeds.view(len(sentence), 1, -1))
        hidden = torch.cat((hidden[-2, :, :], cell[-1, :, :]), dim=1)
        tag_space = self.hidden2tag(hidden)
        tag_scores = fun.log_softmax(tag_space, dim=1)
        return tag_scores


# save data as .yaml files
def generate_yaml_doc(data, yaml_file):
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(data, file)
    file.close()


try:
    # read data
    train_questions_text, train_data = load_file(
        current_path + '/../data/' + config['file_path']['train_data_path'])
except:
    raise SystemExit('Error: Load train data failed! Please check the config file.')
try:
    train_questions_text, dev_data = load_file(
        current_path + '/../data/' + config['file_path']['val_data_path'])
except:
    raise SystemExit('Error: Load validation data failed! Please check the config file.')
try:
    test_questions_text, test_data = load_file(
        current_path + '/../data/' + config['file_path']['test_data_path'])
except:
    raise SystemExit('Error: Load test data failed! Please check the config file.')
distinct_labels = set([row[1] for row in train_data])
test_labels = set([row[1] for row in test_data])


# generate word_to_index dictionary
word_to_ix = generate_word_index(train_questions_text)
label_to_ix = {word: i for i, word in enumerate(distinct_labels)}


# train function
def train_for_bow(model, data, loss_function, optimizer, scheduler):
    train_loss = 0
    train_acc = 0
    predictions = list()
    actual = [row[1] for row in data]
    for i, (sentence, label) in enumerate(data):
        model.zero_grad()

        # If GPU is available, upload the model to GPU
        if use_gpu:
            bow_vec = make_bow_vector(sentence, word_to_ix).cuda()
            target = make_target(label, label_to_ix).cuda()
        else:
            bow_vec = make_bow_vector(sentence, word_to_ix)
            target = make_target(label, label_to_ix)

        # get predictions
        log_probs = model(bow_vec.long())
        loss = loss_function(log_probs, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        predictions.append(make_target_reverse(
            log_probs.argmax(1), label_to_ix))
        train_acc += (log_probs.argmax(1) == target).sum().item()
    scheduler.step()
    # get precision,recall,F1 score and support
    p, r, f, support = precision_recall_fscore_support(
        predictions, actual, average='macro', zero_division=1)
    return train_loss / len(data), train_acc / len(data), p, r, f


def train_for_bilstm(model, data, loss_function, optimizer, scheduler):
    train_loss = 0
    train_acc = 0
    predictions = list()
    actual = [row[1] for row in data]
    for i, (sentence, label) in enumerate(data):
        model.zero_grad()

        # If GPU is available, upload the model to GPU
        if use_gpu:
            bow_vec = prepare_sequence(sentence, word_to_ix).cuda()
            target = prepare_tags(label, label_to_ix).cuda()
        else:
            bow_vec = prepare_sequence(sentence, word_to_ix)
            target = prepare_tags(label, label_to_ix)

        # get predictions
        log_probs = model(bow_vec.long())
        loss = loss_function(log_probs, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        predictions.append(make_target_reverse(
            log_probs.argmax(1), label_to_ix))
        train_acc += (log_probs.argmax(1) == target).sum().item()
    scheduler.step()
    # get precision,recall,F1 score and support
    p, r, f, support = precision_recall_fscore_support(
        predictions, actual, average='macro', zero_division=1)
    return train_loss / len(data), train_acc / len(data), p, r, f


# train function for BOW model
def train_BOW(pretrained_embed, EPOCH, EMBEDDING_DIM, LEARNING_RATE, FREEZE, boost_rate=None, id=''):
    VOCAB_SIZE = len(word_to_ix)
    NUM_LABELS = len(label_to_ix)

    # build a BOW model and set loss function
    model = BOWModeler(VOCAB_SIZE, EMBEDDING_DIM,
                       NUM_LABELS, pretrained_embed, FREEZE)
    loss_function = nn.NLLLoss()

    # if GPU is available, upload the model and loss function to GPU
    if use_gpu:
        model = model.cuda()
        loss_function = loss_function.cuda()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    train_subset = train_data

    # if boost_rate is not None, do boostrapping(used for ensemble model)
    if boost_rate is not None:
        # generate sampler and loader
        train_sampler = torch.utils.data.RandomSampler(train_data, replacement=True, num_samples = int(boost_rate*len(train_data)), generator=None)
        train_subset = torch.utils.data.DataLoader(train_data, batch_size=1,
                                                   sampler=train_sampler)

        # convert dictionary to list
        train_subdata = []
        for i,(a,b) in enumerate(train_subset):
            p = []
            q = []
            for item in a:
                p.append(item[0])
            q.append(p)
            q.append(b[0])
            train_subdata.append(q)
        train_subset = train_subdata

    # train model
    for e in range(EPOCH):
        train_loss, train_acc, train_p, train_r, train_f = train_for_bow(
            model, train_subset, loss_function, optimizer, scheduler)
        valid_loss, valid_acc, valid_p, valid_r, valid_f, _ = test_for_bow(
            model, dev_data, word_to_ix, label_to_ix)

        print('Epoch: %d' % (e + 1))
        print(
            f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\t|\tPrec: {train_p:.4f}(train)\t|\tRec: {train_r:.4f}(train)\t|\tF: {train_f:.4f}(train)\t')
        print(
            f'\tLoss: {valid_loss:.4f}(dev)\t|\tAcc: {valid_acc * 100:.1f}%(dev)\t\t|\tPrec: {valid_p:.4f}(dev)\t|\tRec: {valid_r:.4f}(dev)\t|\tF: {valid_f:.4f}(dev)\t')
        print("\n")

    # save model in .pt file
    save_path = current_path + "/../data/model/" + model_type + id + "_model.pt"
    if os.path.exists(save_path):
        os.remove(save_path)
    try:
        torch.save(model, save_path)
    except:
        raise SystemExit('Error: Save model failed! Please check the file structure.')

    print('Model' + id + ' train complete!')
    return model


# train function for BiLSTM model
def train_BiLSTM(pretrained_embed, EPOCH, EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, FREEZE, boost_rate=None, id=''):
    id = id
    VOCAB_SIZE = len(word_to_ix)
    NUM_LABELS = len(label_to_ix)
    model = BiLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM,
                             VOCAB_SIZE, NUM_LABELS, pretrained_embed, FREEZE)
    loss_function = nn.NLLLoss()

    # if GPU is available, upload the model and loss function to GPU
    if use_gpu:
        model = BiLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM,
                                 VOCAB_SIZE, NUM_LABELS, pretrained_embed, FREEZE).cuda()
        loss_function = nn.NLLLoss().cuda()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    train_subset = train_data

    # if boost_rate is not None, do boostrapping(used for ensemble model)
    if boost_rate is not None:
        # generate sampler and loader
        train_sampler = torch.utils.data.RandomSampler(train_data, replacement=True, num_samples = int(boost_rate*len(train_data)), generator=None)
        train_subset = torch.utils.data.DataLoader(train_data, batch_size=1,
                                                   sampler=train_sampler)

        # convert dictionary to list
        train_subdata = []
        for i,(a,b) in enumerate(train_subset):
            p = []
            q = []
            for item in a:
                p.append(item[0])
            q.append(p)
            q.append(b[0])
            train_subdata.append(q)
        train_subset = train_subdata

    # train model
    for e in range(EPOCH):
        train_loss, train_acc, train_p, train_r, train_f = train_for_bilstm(
            model, train_subset, loss_function, optimizer, scheduler)
        valid_loss, valid_acc, valid_p, valid_r, valid_f,_ = test_for_bilstm(
            model, dev_data, word_to_ix, label_to_ix)

        print('Epoch: %d' % (e + 1))
        print(
            f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\t|\tPrec: {train_p:.4f}(train)\t|\tRec: {train_r:.4f}(train)\t|\tF: {train_f:.4f}(train)\t')
        print(
            f'\tLoss: {valid_loss:.4f}(dev)\t|\tAcc: {valid_acc * 100:.1f}%(dev)\t\t|\tPrec: {valid_p:.4f}(dev)\t|\tRec: {valid_r:.4f}(dev)\t|\tF: {valid_f:.4f}(dev)\t')
        print("\n")

    # save model in .pt file
    save_path = current_path + '/../data/model/' + model_type + id + '_model.pt'
    if os.path.exists(save_path):
        os.remove(save_path)
    try:
        torch.save(model, save_path)
    except:
        raise SystemExit('Error: Save model failed! Please check the file structure.')
    print('Model' + id + ' train complete!')
    return model


# get tags for ensemble model
def get_tag(models, bow_vec_bilstm , bow_vec_bow, ms):
    tags = []
    for i in range(len(models)):
        if ms[i] == 'BiLSTM':
            tag = models[i](bow_vec_bilstm.long()).argmax(1)
        else:
            tag = models[i](bow_vec_bow.long()).argmax(1)
        if use_gpu:
            tag = tag.cpu()
        tags.append(tag)
    label = max(tags, key=tags.count)
    return label


# generate the report of training
def get_report(actual, predictions):
    # get results for each class
    report = classification_report(actual, predictions, output_dict=True, zero_division = 0)
    prediction_table = {}
    # remove the classes not in test set
    for idx in report.keys():
        if idx in test_labels:
            prediction_table[idx] = report[idx]
    # sort the result by precision of each class
    prediction_table = sorted(prediction_table.items(),key=lambda item:item[1]['precision'])
    return prediction_table


# generate the results of testing
def get_res(actual, predictions):
    report = classification_report(actual, predictions, output_dict=True, zero_division = 0)
    prediction_table = {}
    for idx in report.keys():
        if idx not in distinct_labels:
            prediction_table[idx] = report[idx]
    return prediction_table


# function to test model
def test_for_bow(model, data, wti, lti):
    test_loss = 0
    test_acc = 0
    predictions = list()
    actual = [row[1] for row in data]
    loss_function = nn.NLLLoss()
    with torch.no_grad():
        for i, (sentence, label) in enumerate(data):
            if use_gpu:
                bow_vec = make_bow_vector(sentence, wti).cuda()
                target = make_target(label, lti).cuda()
            else:
                bow_vec = make_bow_vector(sentence, wti)
                target = make_target(label, lti)
            log_probs = model(bow_vec.long())
            loss = loss_function(log_probs, target)
            test_loss += loss.item()
            test_acc += (log_probs.argmax(1) == target).sum().item()
            predictions.append(make_target_reverse(log_probs.argmax(1), lti))
    p, r, f, support = precision_recall_fscore_support(
        predictions, actual, average='macro', zero_division=1)
    report = get_report(actual, predictions)

    return test_loss / len(data), test_acc / len(data), p, r, f, report


# function to test model
def test_for_bilstm(model, data, wti, lti):
    test_loss = 0
    test_acc = 0
    predictions = list()
    actual = [row[1] for row in data]
    loss_function = nn.NLLLoss()
    with torch.no_grad():
        for i, (sentence, label) in enumerate(data):
            if use_gpu:
                bow_vec = prepare_sequence(sentence, wti).cuda()
                target = prepare_tags(label, lti).cuda()
            else:
                bow_vec = prepare_sequence(sentence, wti)
                target = prepare_tags(label, lti)

            log_probs = model(bow_vec.long())
            loss = loss_function(log_probs, target)
            test_loss += loss.item()
            test_acc += (log_probs.argmax(1) == target).sum().item()
            predictions.append(make_target_reverse(log_probs.argmax(1), lti))
    p, r, f, support = precision_recall_fscore_support(
        predictions, actual, average='macro', zero_division=1)
    report = get_report(actual, predictions)

    return test_loss / len(data), test_acc / len(data), p, r, f, report


# test function for ensemble model
def test_for_ensemble(models, data, wti, lti):
    labels = []
    targets = []
    ens_models = config['submodel']
    ms = []
    for mk in ens_models.keys():
        ms.append(ens_models[mk]['model_type'])
    with torch.no_grad():
        for i, (sentence, tar) in enumerate(data):
            if use_gpu:
                bow_vec_bilstm = prepare_sequence(sentence, wti).cuda()
                bow_vec_bow = make_bow_vector(sentence, wti).cuda()
            else:
                bow_vec_bilstm = prepare_sequence(sentence, wti)
                bow_vec_bow = make_bow_vector(sentence, wti)

            label = get_tag(models, bow_vec_bilstm , bow_vec_bow, ms)
            label = make_target_reverse(label.item(), lti)
            targets.append(tar)
            labels.append(label)
        right = 0
        for i in range(len(labels)):
            if targets[i] == labels[i]:
                right = right + 1
    report = get_report(np.array(targets), np.array(labels))
    res = get_res(np.array(targets), np.array(labels))
    return right/len(labels), report, res


def check_config(model_t, conf=None):

    if model_t == 'BiLSTM':
        try:
            EPOCH = config['epoch']
            HIDDEN_DIM = config['hidden_dim']
            LEARNING_RATE = config['lr']
            embedding = config['embedding']
            if embedding == 'random':
                EMBEDDING_DIM = config['embedding_dim']
                pretrained_embed = None
                FREEZE = None
            elif embedding == 'pretrained':
                word_embed = load_pretrained_embed(
                    current_path + '/../data/' + config['file_path']['pretrained_path'])
                pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
                EMBEDDING_DIM = len(pretrained_embed[0])
                FREEZE = config['freeze']
        except:
            raise SystemExit('Error: Config file wrong!')
        if HIDDEN_DIM <= 0:
            raise SystemExit('Error: hidden_dim <= 0! Please check the config file.')
        if embedding not in ['random', 'pretrained']:
            raise SystemExit('Error: embedding not in [random,pretrained]! Please check the config file.')
    elif model_t == 'BOW':
        try:
            EPOCH = config['epoch']
            LEARNING_RATE = config['lr']
            embedding = config['embedding']
            if embedding == 'random':
                EMBEDDING_DIM = config['embedding_dim']
                pretrained_embed = None
                FREEZE = None
            elif embedding == 'pretrained':
                word_embed = load_pretrained_embed(
                    current_path + '/../data/' + config['file_path']['pretrained_path'])
                pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
                EMBEDDING_DIM = len(pretrained_embed[0])
                FREEZE = config['freeze']
        except:
            raise SystemExit('Error: Config file wrong!')
        if embedding not in ['random', 'pretrained']:
            raise SystemExit('Error: embedding not in [random,pretrained]! Please check the config file.')
    elif model_t == 'ensemble':
        model_l = config['submodel']
        for sm in model_l.keys():
            if model_l[sm]['model_type'] == 'BiLSTM':
                try:
                    EPOCH = model_l[sm]['epoch']
                    HIDDEN_DIM = model_l[sm]['hidden_dim']
                    LEARNING_RATE = model_l[sm]['lr']
                    embedding = model_l[sm]['embedding']
                    if embedding == 'random':
                        EMBEDDING_DIM = model_l[sm]['embedding_dim']
                        pretrained_embed = None
                        FREEZE = None
                    elif embedding == 'pretrained':
                        word_embed = load_pretrained_embed(
                            current_path + '/../data/' + config['file_path']['pretrained_path'])
                        pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
                        EMBEDDING_DIM = len(pretrained_embed[0])
                        FREEZE = model_l[sm]['freeze']
                except:
                    raise SystemExit('Error: Config file wrong!')
                if HIDDEN_DIM <= 0:
                    raise SystemExit('Error: hidden_dim <= 0! Please check the config file.')
            elif model_l[sm]['model_type'] == 'BOW':
                try:
                    EPOCH = model_l[sm]['epoch']
                    LEARNING_RATE = model_l[sm]['lr']
                    embedding = model_l[sm]['embedding']
                    if embedding == 'random':
                        EMBEDDING_DIM = model_l[sm]['embedding_dim']
                        pretrained_embed = None
                        FREEZE = None
                    elif embedding == 'pretrained':
                        word_embed = load_pretrained_embed(
                            current_path + '/../data/' + config['file_path']['pretrained_path'])
                        pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
                        EMBEDDING_DIM = len(pretrained_embed[0])
                        FREEZE = model_l[sm]['freeze']
                except:
                    raise SystemExit('Error: Config file wrong!')
            else:
                raise SystemExit('Error: Submodel type wrong! Please check the config file.')
            if embedding not in ['random', 'pretrained']:
                raise SystemExit('Error: embedding not in [random,pretrained]! Please check the config file.')
            if EPOCH < 0:
                raise SystemExit('Error: Epoch < 0! Please check the config file.')
            if LEARNING_RATE <= 0:
                raise SystemExit('Error: learning rate <= 0! Please check the config file.')
            if EMBEDDING_DIM <= 0:
                raise SystemExit('Error: embedding_dim <= 0! Please check the config file.')
            if FREEZE not in [None, True, False]:
                raise SystemExit('Error: FREEZE not in [None,True,False]! Please check the config file.')
    if EPOCH < 0:
        raise SystemExit('Error: Epoch < 0! Please check the config file.')
    if LEARNING_RATE <= 0:
        raise SystemExit('Error: learning rate <= 0! Please check the config file.')
    if EMBEDDING_DIM <= 0:
        raise SystemExit('Error: embedding_dim <= 0! Please check the config file.')
    if FREEZE not in [None,True,False]:
        raise SystemExit('Error: FREEZE not in [None,True,False]! Please check the config file.')




# function which gets model type and return the trained model
def get_model(model_t):
    check_config(model_t)

    if model_t == 'BiLSTM':
        EPOCH = config['epoch']
        HIDDEN_DIM = config['hidden_dim']
        LEARNING_RATE = config['lr']
        embedding = config['embedding']
        if embedding == 'random':
            EMBEDDING_DIM = config['embedding_dim']
            pretrained_embed = None
            FREEZE = None
        else:
            word_embed = load_pretrained_embed(
                current_path + '/../data/' + config['file_path']['pretrained_path'])
            pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
            EMBEDDING_DIM = len(pretrained_embed[0])
            FREEZE = config['freeze']
        print('\n')
        print('########## Model Information ##########')
        print('lowercase:' + str(config['preprocess']['lowercase']))
        print('remove_stop_words:' + str(config['preprocess']['remove_stop_words']))
        print('remove_special_characters:' + str(config['preprocess']['remove_special_characters']))
        print('replace_abbreviations:' + str(config['preprocess']['replace_abbreviations']))
        print('remove_white_space:' + str(config['preprocess']['remove_white_space']))
        print('embedding:' + str(embedding))
        print('epoch:' + str(EPOCH))
        print('embedding_dim:' + str(EMBEDDING_DIM))
        print('hidden_dim:' + str(HIDDEN_DIM))
        print('learning_rate:' + str(LEARNING_RATE))
        print('freeze:' + str(FREEZE))
        print('\n')
        print('Start training BiLSTM...')
        print('######### Training Information #########')
        return train_BiLSTM(pretrained_embed, EPOCH, EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, FREEZE)

    elif model_t == 'BOW':
        EPOCH = config['epoch']
        LEARNING_RATE = config['lr']
        embedding = config['embedding']
        if embedding == 'random':
            EMBEDDING_DIM = config['embedding_dim']
            pretrained_embed = None
            FREEZE = None
        else:
            word_embed = load_pretrained_embed(
                current_path + '/../data/' + config['file_path']['pretrained_path'])
            pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
            EMBEDDING_DIM = len(pretrained_embed[0])
            FREEZE = config['freeze']
        print('\n')
        print('########## Model Information ##########')
        print('lowercase:' + str(config['preprocess']['lowercase']))
        print('remove_stop_words:' + str(config['preprocess']['remove_stop_words']))
        print('remove_special_characters:' + str(config['preprocess']['remove_special_characters']))
        print('replace_abbreviations:' + str(config['preprocess']['replace_abbreviations']))
        print('remove_white_space:' + str(config['preprocess']['remove_white_space']))
        print('embedding:' + str(embedding))
        print('epoch:' + str(EPOCH))
        print('embedding_dim:' + str(EMBEDDING_DIM))
        print('learning_rate:' + str(LEARNING_RATE))
        print('freeze:' + str(FREEZE))
        print('\n')
        print('Start training BOW...')
        print('######### Training Information #########')

        return train_BOW(pretrained_embed, EPOCH, EMBEDDING_DIM, LEARNING_RATE, FREEZE)
    elif model_t == 'ensemble':
        models = []
        try:
            model_list = config['submodel']
        except:
            raise SystemExit('Error: Missing submodel information! Please check the config file.')
        try:
            boost_rate = config['boost_rate']
        except:
            raise SystemExit('Error: Missing boost_rate information! Please check the config file.')
        if boost_rate < 0:
            raise SystemExit('Error: boost_rate<0! Please check the config file.')
        id = 1
        for m in model_list.keys():
            m_type = model_list[m]['model_type']
            EPOCH = model_list[m]['epoch']
            if m_type == 'BiLSTM':
                HIDDEN_DIM = model_list[m]['hidden_dim']
            LEARNING_RATE = model_list[m]['lr']
            embedding = model_list[m]['embedding']
            if embedding == 'random':
                EMBEDDING_DIM = model_list[m]['embedding_dim']
                pretrained_embed = None
                FREEZE = None
            else:
                word_embed = load_pretrained_embed(
                    current_path + '/../data/' + config['file_path']['pretrained_path'])
                pretrained_embed = get_pretrained_vec(word_to_ix, word_embed)
                EMBEDDING_DIM = len(pretrained_embed[0])
                FREEZE = model_list[m]['freeze']
            print('\n')
            print('########## Model Information ##########')
            print('model_type:' + str(m_type))
            print('lowercase:' + str(config['preprocess']['lowercase']))
            print('remove_stop_words:' + str(config['preprocess']['remove_stop_words']))
            print('remove_special_characters:' + str(config['preprocess']['remove_special_characters']))
            print('replace_abbreviations:' + str(config['preprocess']['replace_abbreviations']))
            print('remove_white_space:' + str(config['preprocess']['remove_white_space']))
            print('embedding:' + str(embedding))
            print('epoch:' + str(EPOCH))
            print('embedding_dim:' + str(EMBEDDING_DIM))
            if m_type == 'BiLSTM':
                print('hidden_dim:' + str(HIDDEN_DIM))
            print('learning_rate:' + str(LEARNING_RATE))
            print('freeze:' + str(FREEZE))
            print('\n')
            print('Start training ' + str(m) + '...')
            print('######### Training Information #########')
            if m_type == 'BiLSTM':
                smodel = train_BiLSTM(
                    pretrained_embed, EPOCH, EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, FREEZE, boost_rate, str(id))
            elif m_type == 'BOW':
                smodel = train_BOW(pretrained_embed, EPOCH, EMBEDDING_DIM, LEARNING_RATE, FREEZE, boost_rate, str(id))
            id = id + 1
            models.append(smodel)
        return models
    else:
        raise SystemExit('Error: Model type wrong! Please check the config file.')


if __name__ == "__main__":
    try:
        model_type = config['model']
    except:
        raise SystemExit('Error: no model type information in config file!')
    # train and save model
    if args.train:
        word_path = current_path + '/../data/model/' + model_type + '_wti.yaml'
        label_path = current_path + '/../data/model/' + model_type + '_lti.yaml'
        try:
            generate_yaml_doc(word_to_ix, word_path)
            generate_yaml_doc(label_to_ix, label_path)
        except:
            raise SystemExit('Error: Save word_to_idx and label_to_idx failed! Please check the config file.')
        model = get_model(model_type)

    # load and test model
    elif args.test:
        try:
            wti = read_yaml(current_path + '/../data/model/' + model_type + '_wti.yaml')
            lti = read_yaml(current_path + '/../data/model/' + model_type + '_lti.yaml')
        except:
            raise SystemExit('Error: Read word_to_idx and label_to_idx failed! Please check the document structure.')
        if model_type == 'ensemble':
            model_list = config['submodel']
            id = 1
            models = []
            try:
                for e in model_list:
                    model = torch.load(
                        current_path + '/../data/model/ensemble' + str(id) + '_model.pt')
                    model.eval()
                    models.append(model)
                    id = id + 1
            except:
                raise SystemExit('Error: Load model failed! Please check the file structure.')
            acc, report, res = test_for_ensemble(models, test_data, wti, lti)

            result = {}
            result['accuracy'] = acc
            for key in res['weighted avg'].keys():
                result[key] = res['weighted avg'][key]
            print(result)

        elif model_type == 'BiLSTM':
            try:
                model = torch.load(
                    current_path + "/../data/model/" + model_type + "_model.pt")
            except:
                raise SystemExit('Error: Load model failed! Please check the file structure.')
            model.eval()
            test_loss, test_acc, test_p, test_r, test_f, report = test_for_bilstm(
                model, test_data, wti, lti)
            print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)\t|\tPrec: {test_p:.4f}(test)\t|\tRec: {test_r:.4f}(test)\t|\tF: {test_f:.4f}(test)')

        elif model_type == 'BOW':
            try:
                model = torch.load(
                    current_path + "/../data/model/" + model_type + "_model.pt")
            except:
                raise SystemExit('Error: Load model failed! Please check the file structure.')
            model.eval()
            test_loss, test_acc, test_p, test_r, test_f, report = test_for_bow(
                model, test_data, wti, lti)
            print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)\t|\tPrec: {test_p:.4f}(test)\t|\tRec: {test_r:.4f}(test)\t|\tF: {test_f:.4f}(test)')

        else:
            raise SystemExit('Error: Model type wrong! Please check the config file.')
    else:
        raise SystemExit('Error: Command wrong! Please read the readme file.')
