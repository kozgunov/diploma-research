import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')  # applied wikitext as simple example for the beginning
test_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test')

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_node_data(node_id, num_nodes):
    total_size = len(dataset)
    partition_size = total_size // num_nodes # split the text for nodes
    start_idx = node_id * partition_size
    end_idx = start_idx + partition_size if node_id < num_nodes - 1 else total_size
    texts = dataset[start_idx:end_idx]['text']

    data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True) # tokenize the text
        if len(tokens) > 1:
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            data.append((inputs, labels))
    return data

def test_data_loader():  # test text data - we can change it by benchmark
    texts = test_dataset['text']
    data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True) # tokenize again
        if len(tokens) > 1:
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            data.append((inputs, labels))
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataset = data
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader
