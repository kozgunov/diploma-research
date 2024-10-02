import torch
from transformers import LlamaTokenizer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
train_dataset = load_dataset('wikitext', 'wikitext-103-v1',
                             split='train')  # applied wikitext as simple example for the beginning
test_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test')

tokenizer = LlamaTokenizer.from_pretrained('llama-base')


def get_node_data(node_id, num_nodes):
    texts = len(train_dataset)
    num_nodes = 101  # e.g.

    sentences = []  # namely sentences, not text - because we can't stop in the middle of the word/sentence WLOG
    for text in texts:
        sentences.extend(sent_tokenize(text))
    total_sentences = len(sentences)
    partition_size = total_sentences // num_nodes
    start_idx = node_id * partition_size
    end_idx = start_idx + partition_size if node_id < num_nodes - 1 else total_sentences
    node_sentences = sentences[start_idx:end_idx]

    data = []
    for text in node_sentences:
        tokens = tokenizer.encode(text, add_special_tokens=True)  # tokenize the text
        if len(tokens) > 1:
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            data.append((inputs, labels))
    return data


def test_data_loader():  # test text data - we can change it by benchmark
    texts = test_dataset['text']
    data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)  # tokenize again
        if len(tokens) > 1:
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            data.append((inputs, labels))
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataset = data
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader


def load_and_prepar1e_data():
    datasets = []
    # more datasets or unify them
    datasets.append(load_dataset('wikipedia', '20220301.en', split='train'))
    datasets.append(load_dataset('bookcorpus', split='train'))
    datasets.append(load_dataset('openwebtext', split='train'))
    combined_dataset = datasets[0]  # combine & shuffle
    for ds in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(ds)
    combined_dataset = combined_dataset.shuffle(seed=42)
    return combined_dataset


def load_and_prepare_data():
    glue_dataset = load_dataset('glue', 'sst2')  # load the GLUE benchmark dataset
    train_dataset = glue_dataset['train']  # split the dataset
    test_dataset = glue_dataset['validation']
    return train_dataset, test_dataset
