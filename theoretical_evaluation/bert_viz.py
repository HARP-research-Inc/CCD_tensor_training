from transformers import BertModel, BertTokenizer
from bertviz import head_view
import torch
import os

model_name = "sentence-transformers/bert-base-nli-mean-tokens"
model = BertModel.from_pretrained(model_name, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_name)

sentence = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
outputs = model(**inputs)

# Keep attention as PyTorch tensors, just move to CPU
attention = [layer_attn.detach().cpu() for layer_attn in outputs.attentions]

# Get the visualization HTML
viz_html = head_view(
    attention=attention,
    tokens=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
)

# Save to HTML file
with open('bert_attention.html', 'w', encoding='utf-8') as f:
    f.write(str(viz_html))

print("Visualization saved to bert_attention.html")
