import copy
import random
import main
import torch
import numpy as np
from diploma.data_preparation import tokenizer
from node import Node, encrypt_updates
import re
from opacus import PrivacyEngine


class Attacks_Simulation:
    def __init__(self, nodes):
        self.nodes = nodes


# data poisoning
async def local_training(self, epochs=5):
    self.initialize_model_state()
    self.model.train()
    for epoch in range(epochs):
        for inputs, labels in self.data_loader():
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
    delta_w = self.get_model_updates()
    if self.is_malicious:  # poisoned here
        for name in delta_w:
            delta_w[name] = delta_w[name] * 10  # exaggerate the updates
    encrypted_updates = encrypt_updates(delta_w)
    return encrypted_updates

    # generate poisoned smth to check how will it go on



# sybil attack
def sybil_attack(nodes, sybil_nodes_count):
    malicious_node = nodes[0]  # assume node is malicious
    for _ in range(sybil_nodes_count):
        new_node = copy.deepcopy(malicious_node)
        new_node.node_id = generate_new_node_id(nodes)
        nodes.append(new_node)


def generate_new_node_id(existing_nodes):
    existing_ids = {node.node_id for node in existing_nodes}
    new_id = max(existing_ids) + 1
    return new_id

# sanitization of the whole data
def sanitize_data(data):
    sanitized_data = []
    for inputs, labels in data:
        if is_valid_input(inputs) and is_valid_label(labels):
            sanitized_data.append((inputs, labels))
    return sanitized_data

def is_valid_input(inputs):
    return torch.all(inputs >= tokenizer.vocab_size)

def is_valid_label(labels):
    return labels.item() in [0, 1]


# detection of outliers
def detect_outliers(data):
    lengths = [len(inputs) for inputs, _ in data]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    threshold = mean_length + 3 * std_length
    filtered_data = [(inputs, labels) for inputs, labels in data if len(inputs) <= threshold]
    return filtered_data

# remove redudency
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for inputs, labels in data:
        data_hash = hash((tuple(inputs.tolist()), labels.item()))
        if data_hash not in seen:
            seen.add(data_hash)
            unique_data.append((inputs, labels))
    return unique_data


# mutual  preprocessing, which will save of personal data (if model, will know them exactly, then it can say so)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess_dataset(dataset):
    preprocessed_data = []
    for example in dataset:
        text = preprocess_text(example['text'])
        tokens = tokenizer.encode(text, add_special_tokens=True)
        inputs = torch.tensor(tokens[:-1])
        labels = torch.tensor(tokens[1:])
        preprocessed_data.append((inputs, labels))
    return preprocessed_data


def adversarial_training(model, data_loader, optimizer, loss_fn):
    for inputs, labels in data_loader:
        inputs_adv = generate_adversarial_examples(inputs, labels, model) #  combine original and adversarial inputs
        combined_inputs = torch.cat([inputs, inputs_adv])
        combined_labels = torch.cat([labels, labels])
        optimizer.zero_grad() # training step
        outputs = model(combined_inputs)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), combined_labels.view(-1))
        loss.backward()
        optimizer.step()

def generate_adversarial_examples(inputs, labels, model):
    pass

# differential privacy
def train_with_privacy(model, data_loader, optimizer):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

def loss_fn(first_param, second_param):
    return True










def evasion_attack(node):
    pass

def backdoor_attack(node):
    pass


def malicious_dataset_attack(node):
    node.data = generate_malicious_data()


def generate_malicious_data():
    malicious_data = []
    return malicious_data


def model_poisoning_attack(node):  # modify the node's model updates to poison the global model
    original_get_model_updates = node.get_model_updates

    node.get_model_updates = poisoned_get_model_updates(original_get_model_updates)


def poisoned_get_model_updates(original_get_model_updates):
    delta_w = original_get_model_updates()
    for name in delta_w:
        delta_w[name] += torch.randn(delta_w[name].shape) * 10  # exaggerated noise
    return delta_w


async def attack_nodes(nodes):
    malicious_node_probability = int(0.1 * main.num_nodes)
    for node in nodes:
        node.is_malicious = random.random() < malicious_node_probability

