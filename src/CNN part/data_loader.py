import numpy as np
import re
import itertools
from collections import Counter
import  json


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_json_raw_data(chunk_path, num_limit):
    x_text = []
    y=[]
    count = [0, 0, 0]
    with open(chunk_path, 'r') as json_chunk_file:
        for lines in json_chunk_file:
            # line_buf = json_chunk_file.readline()
            line_jsonify = json.loads(lines)  # build a dictionary

            helpful_list = line_jsonify['helpful']

            # if helpful_list[1] == 0 or helpful_list[0] == helpful_list[1] / 2:
            if helpful_list[0] == helpful_list[1] / 2:
                helpful_score = [0, 1, 0]  # Neutral
                count[1] += 1
                if count[1]>num_limit:
                    continue
            elif helpful_list[0] > helpful_list[1] / 2:
                helpful_score = [0, 0, 1]  # helpful
                count[2] += 1
                if count[2]>num_limit:
                    continue
            elif helpful_list[0] < helpful_list[1] / 2:
                helpful_score = [1, 0, 0]  # not helpful
                count[0] += 1
                if count[0]>num_limit:
                    continue
            x_text.append(str(line_jsonify['reviewText']))
            y.append(helpful_score)

        json_chunk_file.close()
    # print(self.review_text_list[6])
    # print(self.review_text_list[7])
    print('Chunk loaded. Label count: ')
    print(count)
    return [x_text, np.array(y)]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors