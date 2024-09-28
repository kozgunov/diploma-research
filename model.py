from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class GPT2Model(nn.Module):
    def __init__(self):
        super(GPT2Model, self).__init__()
        config = GPT2Config(
            n_embd=128,
            n_layer=4,
            n_head=4,
            vocab_size=50257
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        return logits


def evaluate_global_model(model):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, labels in test_data_loader():
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, -1)
            total_correct += (predicted == labels).sum().item()
            total_tokens += labels.numel()
    perplexity = np.exp(total_loss / total_tokens)
    accuracy = total_correct / total_tokens * 100
    print(f'Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.2f}%')


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.inputs.append(torch.tensor(tokens[:-1]))
            self.labels.append(torch.tensor(tokens[1:]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def get_node_data(node_id): # partition the dataset
    node_texts = partitioned_texts[node_id]
    dataset = WikiTextDataset(node_texts, tokenizer)
    return dataset

def test_data_loader(): # create data loader for the test dataset
    dataset = WikiTextDataset(test_texts, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader
