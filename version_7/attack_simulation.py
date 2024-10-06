from transformers import LlamaTokenizer
import copy
import random
import torch
import numpy as np
import re
from node import Node
from torch.utils.data import DataLoader
from torch import nn
from opacus import PrivacyEngine

print('NOW, you are in attack_simulation.py')



tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b')


class AttacksSimulation:
    def __init__(self, nodes):
        self.nodes = nodes # only the nodes are indeed for such simulations
        print('NOW, you are in AttacksSimulation in attack_simulation.py')

    def assign_malicious_nodes(self, malicious_node_probability=0.1): # randomly assigns nodes as malicious for check
        print('NOW, you are in assign_malicious_nodes in attack_simulation.py')
        for node in self.nodes:
            node.is_malicious = random.random() < malicious_node_probability

    def simulate_data_poisoning(self): # simulates data poisoning attack by modifying local training data of such node
        print('NOW, you are in simulate_data_poisoning in attack_simulation.py')
        for node in self.nodes:
            if node.is_malicious: # poison the node's local data
                node.data = self.generate_malicious_data(node.data)
                print(f"Node {node.node_id} data has been poisoned.")

    def generate_malicious_data(self, data): # generates malicious data to poison the training dataset
        print('NOW, you are in generate_malicious_data in attack_simulation.py')
        poisoned_data = []
        for inputs, labels in data: # introduce incorrect labels or perturb inputs
            poisoned_labels = labels.clone()
            poisoned_labels = (poisoned_labels + 1) % tokenizer.vocab_size  # shift labels to incorrect tokens
            poisoned_data.append((inputs, poisoned_labels))
        return poisoned_data

    def simulate_sybil_attack(self, sybil_nodes_count): # Sybil attack creates duplicate nodes
        print('NOW, you are in simulate_sybil_attack in attack_simulation.py')
        malicious_nodes = [node for node in self.nodes if node.is_malicious]
        for malicious_node in malicious_nodes:
            for _ in range(sybil_nodes_count):
                new_node = copy.deepcopy(malicious_node)
                new_node.node_id = self.generate_new_node_id()
                self.nodes.append(new_node)
                print(f"Sybil node {new_node.node_id} created.")

    def generate_new_node_id(self): # generates a unique node ID
        print('NOW, you are in generate_new_node_id in attack_simulation.py')
        existing_ids = {node.node_id for node in self.nodes}
        new_id = max(existing_ids) + 1
        return new_id

    def simulate_evasion_attack(self): # simulates evasion attack (malicious nodes craft updates to evade detection)
        print('NOW, you are in simulate_evasion_attack in attack_simulation.py')
        for node in self.nodes:
            if node.is_malicious:
                node.perform_evasion_attack = True  # set flag to modify behavior during the training

    def simulate_backdoor_attack(self): # simulates backdoor attack by injecting triggers into the model updates (specialized and very advanced attack)
        print('NOW, you are in simulate_backdoor_attack in attack_simulation.py')
        for node in self.nodes:
            if node.is_malicious:
                node.perform_backdoor_attack = True  # set a flag to modify behavior during the training

    def sanitize_data(self): # apply data sanitization methods to all nodes' data
        print('NOW, you are in sanitize_data in attack_simulation.py')
        for node in self.nodes:
            node.data = self.remove_duplicates(self.detect_outliers(self.sanitize_node_data(node.data)))

    def sanitize_node_data(self, data): # sanitizes node's data by validating inputs and labels
        print('NOW, you are in sanitize_node_data in attack_simulation.py')
        sanitized_data = []
        for inputs, labels in data:
            if self.is_valid_input(inputs) and self.is_valid_label(labels):
                sanitized_data.append((inputs, labels))
        return sanitized_data

    def is_valid_input(self, inputs): # checks if inputs are valid
        print('NOW, you are in is_valid_input in attack_simulation.py')
        return torch.all(inputs < tokenizer.vocab_size) # ensure all tokens are less than the tokenizer's vocab size

    def is_valid_label(self, labels): # check if labels are valid
        print('NOW, you are in is_valid_label in attack_simulation.py')
        return torch.all(labels >= 0) and torch.all(labels < tokenizer.vocab_size) # for classification tasks, labels should be within valid range

    def detect_outliers(self, data): # detects and removes outliers from the data based on input length
        print('NOW, you are in detect_outliers in attack_simulation.py')
        lengths = [len(inputs) for inputs, _ in data]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        threshold = mean_length + 3 * std_length
        filtered_data = [(inputs, labels) for inputs, labels in data if len(inputs) <= threshold]
        return filtered_data

    def remove_duplicates(self, data): # removes duplicate entries from the data
        print('NOW, you are in remove_duplicates in attack_simulation.py')
        seen = set()
        unique_data = []
        for inputs, labels in data:
            data_hash = hash((tuple(inputs.tolist()), tuple(labels.tolist())))
            if data_hash not in seen:
                seen.add(data_hash)
                unique_data.append((inputs, labels))
        return unique_data # duplicated data will never get here

    def preprocess_text(self, text): # preprocesses text (ofc, unpersonalization is important here for any tasks)
        print('NOW, you are in preprocess_text in attack_simulation.py')
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def preprocess_dataset(self, dataset): # application preprocessing to the dataset
        print('NOW, you are in preprocess_dataset in attack_simulation.py')
        preprocessed_data = []
        for example in dataset:
            text = self.preprocess_text(example['text'])
            tokens = tokenizer.encode(text, add_special_tokens=True)
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            preprocessed_data.append((inputs, labels))
        return preprocessed_data

    def adversarial_training(self, model, data_loader, optimizer, loss_fn): # trains the model using adversarial example
        print('NOW, you are in adversarial_training in attack_simulation.py')
        for inputs, labels in data_loader:
            inputs_adv = self.generate_adversarial_examples(inputs, labels, model) # generate adversarial examples
            combined_inputs = torch.cat([inputs, inputs_adv]) # combine original and adversarial inputs
            combined_labels = torch.cat([labels, labels])
            optimizer.zero_grad()
            outputs = model(combined_inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), combined_labels.view(-1))
            loss.backward()
            optimizer.step()

    def generate_adversarial_examples(self, inputs, labels, model): # generates adversarial example and implement adversarial example generation
        print('NOW, you are in generate_adversarial_examples in attack_simulation.py')
        inputs_adv = inputs.clone().detach()
        inputs_adv.requires_grad = True

        outputs = model(inputs_adv)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        model.zero_grad()
        loss.backward()

        epsilon = 0.01
        perturbation = epsilon * inputs_adv.grad.sign() # apply perturbation
        inputs_adv = inputs_adv + perturbation
        inputs_adv = inputs_adv.detach()
        return inputs_adv

    def train_with_privacy(self, model, data_loader, optimizer): # train the model with differential privacy
        print('NOW, you are in train_with_privacy in attack_simulation.py')

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = (privacy_engine.make_private
        (
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        ))

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
