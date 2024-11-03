from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
import optuna

print('NOW, you are  in models.py')


class LlamaModel(nn.Module):
    def __init__(self):
        super(LlamaModel, self).__init__() # load the pre-trained LLaMA-2 model for transfer learning
        self.model = LlamaForCausalLM.from_pretrained('facebook/llama-2-7b') # transfer learning method implementation
        self.tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b') # corresponding tokenizator to llama2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device to GPU (obv.)
        self.model.to(self.device) # same
        super(LlamaModel, self).__init__()

        #config = LlamaConfig(hidden_size=512,num_hidden_layers=6,num_attention_heads=8,intermediate_size=2048,vocab_size=32000)

    def forward(self, input_ids, attention_mask=None):
        print('NOW, you are in forward in models.py')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) # perform a forward pass through the model
        logits = outputs.logits # got non-normalized predictions
        return logits


