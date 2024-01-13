from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Encode text
text = "导航到上海交通大学足球场，我们"  # Example text in Chinese
encoded_input = tokenizer(text, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print(tokens)

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-chinese')

# Forward pass, get hidden states
with torch.no_grad():
    output = model(**encoded_input)

# Get the embeddings for the input text
embeddings = output.last_hidden_state
print(embeddings.shape)

# embeddings now contains the embeddings for each token in the input
