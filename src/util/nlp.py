import collections

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def make_vocab(dataset, max_vocab_size=20000, min_freq=2):
    counts = collections.defaultdict(int)
    for tokens, _ in dataset:
        for token in tokens:
            counts[token] += 1

    vocab = {'<eos>': 0, '<unk>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return np.array(ids, 'i')


def transform_to_array(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(tokens, vocab), np.array([cls], np.float32))
                for tokens, cls in dataset]
    else:
        return [make_array(tokens, vocab)
                for tokens in dataset]


def make_train_test_dataset(tag_and_element_list):

    ranks = [i for i in range(len(tag_and_element_list))]
    train, test, train_rank, test_rank = train_test_split(
        tag_and_element_list, ranks, test_size=0.1, random_state=0)

    train_rank = np.asarray(train_rank).reshape(-1, 1)
    test_rank = np.asarray(test_rank).reshape(-1, 1)

    # import pdb
    # pdb.set_trace()

    mms = MinMaxScaler()
    train_rank = mms.fit_transform(train_rank)
    test_rank = mms.transform(test_rank)

    train = [(t, r) for t, r in zip(train, train_rank)]
    test = [(t, r) for t, r in zip(test, test_rank)]

    vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab
