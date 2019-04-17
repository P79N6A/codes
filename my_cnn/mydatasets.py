# coding:utf-8
import os
import random
import json
from torchtext import data

random.seed(0)


class Vulgar(data.Dataset):
    dirname = 'vulgar_data'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, sample_num=None, examples=None, **kwargs):
        """Create an dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        def clean_str(string):
            """
            Tokenization/string cleaning for datasets.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'vulgar.neg'), errors='ignore') as f:
                for i, line in enumerate(f):
                    # if sample_num and i == sample_num:
                    #     break
                    if i == 10000:
                        break
                    line = line.strip()
                    json_data = json.loads(line)
                    text = json_data.get('data', '')
                    examples.append(data.Example.fromlist([text, 0], fields))
            with open(os.path.join(path, 'vulgar.pos'), errors='ignore') as f:
                for i, line in enumerate(f):
                    # if sample_num and i == sample_num:
                    #     break
                    if i == 1000:
                        break
                    line = line.strip()
                    json_data = json.loads(line)
                    text = json_data.get('data', '')
                    examples.append(data.Example.fromlist([text, 1], fields))
        super(Vulgar, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, sample_num=None, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(text_field, label_field, path=None, sample_num=sample_num, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio * len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


'''
1. 每个batch正例和负例一样多。正例中选50个，负例中选50个（不放回选）。
2. 修改损失函数，提高正例的比例。
'''
