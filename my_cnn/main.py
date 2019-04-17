# coding:utf-8
import os
import logging
import datetime
import time
import argparse
import torchtext.data as data
import torchtext.vocab as vocab
import torchtext.datasets as datasets
import mydatasets
import torch
import model
import train
import torch.nn as nn

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-log', type=str, default=str(int(time.time())), help='')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=1, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=10000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-sample-num', type=int, default=300, help='neg and pos seperate sample num')
parser.add_argument('-load-vec', type=str, default='', help='word2vec file')
parser.add_argument('-pos-weight', type=float, default=0.5, help='default: 0.5 for unbalanced label')
args = parser.parse_args()
logging.basicConfig(level=logging.ERROR, filename=args.log + '.log',
                    format='%(asctime)s %(levelname)-5s%(message)s %(filename)s:%(lineno)d')


# load Vulgar dataset
def vulgar(text_field, label_field, args, **kargs):
    sample_num = args.sample_num
    train_data, dev_data = mydatasets.Vulgar.splits(text_field, label_field, sample_num=sample_num)
    logging.critical('train_data:%s, dev_data:%s', len(train_data.examples), len(dev_data.examples))
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    logging.critical('total data dist: %s', label_field.vocab.freqs)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kargs)
    total_steps = int(len(train_data.examples) / args.batch_size) + 1
    return train_iter, dev_iter, total_steps


def main():
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    logging.critical('starting loading data')
    train_iter, dev_iter, total_steps = vulgar(text_field, label_field, args, device=-1, repeat=False)
    if args.load_vec:
        if args.load_vec == 'hi':
            args.load_vec = 'model/hi_1105_ml_100.w2v'

        logging.critical('start load word2vec')
        embeddings_file = args.load_vec
        vectors = vocab.Vectors(embeddings_file)
        text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_field.vocab.vectors))
        args.embed_dim = vectors.dim
        embedding.weight.requires_grad = True
        # logging.critical(embedding.weight.requires_grad)
    else:
        # update args and print
        args.embed_num = len(text_field.vocab)
        embedding = nn.Embedding(args.embed_num, args.embed_dim)

    args.class_num = len(label_field.vocab) - 1  # 有个<unk>
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]  # args中-变成了_
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging.critical('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        logging.critical("\t{}={}".format(attr.upper(), value))
    # model
    cnn = model.CNN_Text(args, embedding)
    if args.snapshot is not None:
        logging.critical('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))
    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    try:
        train.train(train_iter, dev_iter, cnn, args, total_steps)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    main()
