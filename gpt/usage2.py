import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Encode some inputs
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
indexed_tokens_1 = tokenizer.encode(text_1)
indexed_tokens_2 = tokenizer.encode(text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])



# Load pre-trained model (weights)
model = GPT2Model.from_pretrained('gpt2')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    hidden_states_1, past = model(tokens_tensor_1)
    # past can be used to reuse precomputed hidden state in a subsequent predictions
    # (see beam-search examples in the run_gpt2.py example).
    hidden_states_2, past = model(tokens_tensor_2, past=past)


# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()


# Predict all tokens
with torch.no_grad():
    predictions_1, past = model(tokens_tensor_1)
    # past can be used to reuse precomputed hidden state in a subsequent predictions
    # (see beam-search examples in the run_gpt2.py example).
    predictions_2, past = model(tokens_tensor_2, past=past)

# get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.decode([predicted_index])
print(predicted_token)