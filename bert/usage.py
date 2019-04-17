# coding:utf-8
import torch
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# word_dict: /Users/admin/.pytorch_pretrained_bert/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084

name = 'bert-base-multilingual-cased'
# name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(name)

text = "[CLS] who was Jim Henson? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
'''
['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '[SEP]']
'''

masked_index = 8
tokenized_text[masked_index] = '[MASK]'

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([segments_ids])

model = BertModel.from_pretrained(name)
model.eval()

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensor)
print(encoded_layers)

model = BertForMaskedLM.from_pretrained(name)
model.eval()
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensor)
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print(predicted_token)
