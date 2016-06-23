import numpy as np


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples1 = []
    positive_examples2 = []
    negative_examples1 = []
    negative_examples2 = []
    with open('./data/positive.txt') as f:
        for i in f.readlines():
            item1, item2 = i.split('\t')
            positive_examples1.append(item1)
            positive_examples2.append(item2)

    with open('./data/negative.txt') as f:
        for i in f.readlines():
            item1, item2 = i.split('\t')
            negative_examples1.append(item1)
            negative_examples2.append(item2)

    # Split by words
    x_text1 = positive_examples1 + negative_examples1
    x_text2 = positive_examples2 + negative_examples2

    # Generate labels
    positive_labels1 = [[0, 1] for _ in positive_examples1]
    positive_labels2 = [[0, 1] for _ in positive_examples2]
    negative_labels1 = [[1, 0] for _ in negative_examples1]
    negative_labels2 = [[1, 0] for _ in negative_examples2]
    y = np.concatenate([positive_labels1, negative_labels1, positive_labels2, negative_labels2], 0)
    return [x_text1, x_text2, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
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

