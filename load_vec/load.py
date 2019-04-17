import torch
import torch.nn as nn
import torchtext.data as data
import torchtext.vocab as vocab

path_to_embeddings_file = '/Users/admin/Documents/repos/codes/my_cnn/model/hi_1105_ml_100.w2v'
# use torchtext to define the dataset field containing text
text_field = data.Field(sequential=True)

# load your dataset using torchtext, e.g.
dataset = data.Dataset(examples=..., fields=[('text', text_field), ...])

# build vocabulary
text_field.build_vocab(dataset)

# I use embeddings created with
# model = gensim.models.Word2Vec(...)
# model.wv.save_word2vec_format(path_to_embeddings_file)

# load embeddings using torchtext
vectors = vocab.Vectors(path_to_embeddings_file)  # file created by gensim
text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

# when defining your network you can then use the method mentioned by blue-phoenox
embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_field.vocab.vectors))

# pass data to the layer
dataset_iter = data.Iterator(dataset, ...)
for batch in dataset_iter:
    ...
    embedding(batch.text)
