import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class LSTM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        self.n_vocab = n_vocab
        self.n_units = n_units

        super(LSTM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, self.n_units, ignore_label=-1)
            self.lstm = L.NStepLSTM(n_layers=1,
                                    in_size=self.n_units,
                                    out_size=self.n_units,
                                    dropout=0.2)
            self.fc = L.Linear(self.n_units, 1)

    def __call__(self, xs):

        h = sequence_embed(self.embed, xs)
        hx, cx, ys = self.lstm(None, None, h)

        h = self.fc(ys)

        return h
