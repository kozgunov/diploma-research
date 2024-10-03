import torch
from transformers import LlamaTokenizer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

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
    loader = DataLoader(data, batch_size=32, shuffle=False)
    return loader


def load_and_prepar1e_data():  # more datasets or unify them - 0th datasets

    datasets = [load_dataset('wikipedia', '20220301.en', split='train'), load_dataset('bookcorpus', split='train'),
                load_dataset('openwebtext', split='train')]
    combined_dataset = datasets[0]  # combine & shuffle
    for ds in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(ds)
    combined_dataset = combined_dataset.shuffle(seed=42)
    return combined_dataset


# task_names: 'mnli', 'qqp', 'sst2', 'boolq', 'rte', 'cb', 'copa', 'multirc', 'record', 'wic',  'wsc', 'ax-b and ax-g', 'qnli', 'mrpc', 'cola' - processes are going automatically

def load_and_prepare_glue(task_name):  # load the SuperGLUE dataset - 1st benchmark
    glue_dataset = load_dataset('glue', task_name)  # load the GLUE benchmark dataset
    train_dataset = glue_dataset['train']  # split the dataset
    test_dataset = glue_dataset['validation']
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return train_dataset, test_dataset

def preprocess_function(task_name, examples):  # that's for check before integration into the real implementation
    if task_name in ['mnli', 'rte']: # entailment tasks
        texts = (examples['premise'], examples['hypothesis'])
    elif task_name == 'qqp': # paraphrase tasks
        texts = (examples['question1'], examples['question2'])
    elif task_name == 'sst2': # sentiment analysis
        texts = examples['sentence']
    else:
        raise ValueError(f"Task {task_name} not supported.")

    result = tokenizer(*texts, padding='max_length', max_length=512, truncation=True)
    return result


def load_and_prepare_superglue(task_name):  # load the SuperGLUE dataset - 2nd benchmark

    superglue_dataset = load_dataset('super_glue', task_name)

    train_dataset = superglue_dataset['train']
    validation_dataset = superglue_dataset['validation']
    test_dataset = superglue_dataset['test']

    train_data = preprocess_superglue_dataset(train_dataset)
    validation_data = preprocess_superglue_dataset(validation_dataset)
    test_data = preprocess_superglue_dataset(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return train_data, validation_data, test_data


def preprocess_superglue_dataset(dataset):
    tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b')

    data = []
    for example in dataset: # for 'boolq' task

        question = example['question']
        passage = example['passage']
        label = example['label']  # 0 or 1

        input_text = f"Question: {question} Context: {passage}"
        tokens = tokenizer.encode(input_text, add_special_tokens=True)
        inputs = torch.tensor(tokens)
        labels = torch.tensor(label)
        data.append((inputs, labels))
    return data



