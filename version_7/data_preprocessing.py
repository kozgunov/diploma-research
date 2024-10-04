import torch
from transformers import LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize

print('NOW, you are in data_preprocession.py')

nltk.download('punkt')
train_dataset = load_dataset('wikitext', 'wikitext-103-v1',
                             split='train')  # load wikitext as simple example for the beginning
test_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test')

# tokenizer = LlamaTokenizer.from_pretrained('llama-base') we  can use this tokenizer as default (or may be they are the same)
tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b')

# task_names: 'mnli', 'qqp', 'sst2', 'boolq', 'rte', 'cb', 'copa', 'multirc', 'record', 'wic',  'wsc', 'ax-b and ax-g', 'qnli', 'mrpc', 'cola' - processes are going automatically

print('NOW, tokenizer, wikitexts are loaded successfully')


def get_node_data(node_id, num_nodes):  # function for partition data among nodes
    print('NOW, you are in get_node_data in data_preprocession.py')

    num_nodes = 101  # e.g.
    texts = train_dataset['text']  # get the text from the dataset(s)

    sentences = []  # namely sentences, not text - because we can't stop in the middle of the word/sentence WLOG
    for text in texts:
        sentences.extend(sent_tokenize(text))  # tokenization of the text
    total_sentences = len(sentences)  # sentence's distribution among nodes
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
    print('NOW, you are in test_data_loader in data_preprocession.py')
    texts = test_dataset['text']
    data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)  # tokenize again
        if len(tokens) > 1:
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            data.append((inputs, labels))
    from torch.utils.data import DataLoader  # import this entity namely here
    loader = DataLoader(data, batch_size=32,
                        shuffle=False)  # create loader, which will load the future datasets semi-automatically
    return loader


def load_and_prepare_data():  # more datasets or unify them - 0th datasets
    print('NOW, you are in load_and_prepare_data in data_preprocession.py')

    datasets = [load_dataset('wikipedia', '20220301.en', split='train'), load_dataset('bookcorpus', split='train'),
                load_dataset('openwebtext', split='train')]
    combined_dataset = datasets[0]  # combine & shuffle
    for ds in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(ds)
    combined_dataset = combined_dataset.shuffle(seed=42)
    return combined_dataset


def load_and_prepare_glue(task_name):  # load the GLUE dataset - 1st benchmark
    print('NOW, you are in load_and_prepare_glue in data_preprocession.py')
    glue_dataset = load_dataset('glue', task_name)  # load the GLUE benchmark dataset
    train_dataset = glue_dataset['train']  # split the dataset
    test_dataset = glue_dataset['validation']
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') - or we can use llama.tokenizar
    train_data = preprocess_glue_dataset(train_dataset, task_name)
    test_data = preprocess_glue_dataset(test_dataset, task_name)
    return train_data, test_data


def preprocess_glue_dataset(dataset, task_name):  # provided 3 examples for proposed tasks and respectively preprocessing
    print('NOW, you are in preprocess_glue_dataset in data_preprocession.py')
    data = []
    for example in dataset:
        if task_name in ['mnli', 'rte']:  # entailment tasks
            texts = (example['premise'], example['hypothesis'])
        elif task_name == 'qqp':  # paraphrase tasks
            texts = (example['question1'], example['question2'])
        elif task_name == 'sst2':  # sentiment analysis
            texts = (example['sentence'],)
        else:
            raise ValueError(f"Task {task_name} not supported.")

        # Tokenize the texts
        tokens = tokenizer.encode_plus(*texts, padding='max_length', add_special_tokens=True, max_length=512,
                                       truncation=True)
        inputs = torch.tensor(tokens['input_ids'])
        labels = torch.tensor(example['label'])
        data.append((inputs, labels))
    return data


def load_and_prepare_superglue(task_name):  # load the SuperGLUE dataset - 2nd benchmark
    print('NOW, you are in load_and_prepare_superglue in data_preprocession.py')

    superglue_dataset = load_dataset('super_glue', task_name)  # load dataset

    train_dataset = superglue_dataset['train']
    validation_dataset = superglue_dataset['validation']
    test_dataset = superglue_dataset['test']

    train_data = preprocess_superglue_dataset(train_dataset, task_name)
    validation_data = preprocess_superglue_dataset(validation_dataset, task_name)
    test_data = preprocess_superglue_dataset(test_dataset, task_name)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') - also we can subtitute by llama.tokenizer
    return train_data, validation_data, test_data


def preprocess_superglue_dataset(dataset, task_name):  # preprocessing the 2nd dataset w.r.t. proposed task
    print('NOW, you are in preprocess_superglue_dataset in data_preprocession.py')
    tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b')
    input_text = ' '
    data = []
    for example in dataset:  # for 'boolq' task as instance
        if task_name == 'boolq':
            question = example['question']
            passage = example['passage']
            label = example['label']  # 0 or 1
            input_text = f"Question: {question} Context: {passage}"
        else:
            raise ValueError(f"Task {task_name} not supported.")

        tokens = tokenizer.encode(input_text, add_special_tokens=True, max_length=512, truncation=True)
        inputs = torch.tensor(tokens)
        labels = torch.tensor(label)
        data.append((inputs, labels))
    return data
