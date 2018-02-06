import argparse
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert

from model import LSTM
from util.nlp import make_train_test_dataset


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of LSTM units in each layer')

    return parser.parse_args()


def main():

    args = parse_args()

    tag_and_element_file = Path('.') / 'tag_and_element.txt'
    with tag_and_element_file.open('r') as rf:
        tag_and_element_list = [line.strip().split() for line in rf.readlines()]

    train, test, vocab = make_train_test_dataset(tag_and_element_list)

    # import pdb
    # pdb.set_trace()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    model = L.Classifier(LSTM(n_vocab=len(vocab), n_units=args.unit),
                         lossfun=F.mean_squared_error)
    model.compute_accuracy = False

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=lambda batch, device: convert.concat_examples(
                                           batch, device, padding=-1))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        converter=lambda batch, device: convert.concat_examples(
                                            batch, device, padding=-1)))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
