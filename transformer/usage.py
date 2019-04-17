import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

# Tokenized input
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)

# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Load pre-trained model (weights)
model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
model.eval()

with torch.no_grad():
    # Predict hidden states features for each layer
    hidden_states_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

# Load pre-trained model (weights)
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
model.eval()


with torch.no_grad():
    # Predict all tokens
    predictions_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

# get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# assert predicted_token == 'who'