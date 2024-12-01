from transformers import LlamaTokenizer
import copy
import random
import torch
import numpy as np
import re
from opacus import PrivacyEngine
from torch import nn

print('NOW, you are in attack_simulation.py')

token = "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"
tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b', use_auth_token=token)

class AttacksSimulation:
    def __init__(self, nodes):
        print('NOW, you are in __init__ in AttacksSimulation')
        self.nodes = nodes

    def assign_malicious_nodes(self, malicious_node_probability=0.1):
        """Randomly assigns nodes as malicious and assigns unique attacks."""
        print('NOW, you are in assign_malicious_nodes in attack_simulation.py')
        attack_types = [
            "data_poisoning",
            "sybil_attack",
            "evasion_attack",
            "backdoor_attack",
            "model_inversion_attack",
            "membership_inference_attack"
        ]
        
        random.shuffle(attack_types)
        attack_index = 0
        
        for node in self.nodes:
            if random.random() < malicious_node_probability:
                node.is_malicious = True
                attack = attack_types[attack_index]
                print(f"Node {node.node_id} assigned as malicious with attack type: {attack}")
                self.assign_attack(node, attack)
                attack_index = (attack_index + 1) % len(attack_types)

    def assign_attack(self, node, attack_type):
        """Assign a specific attack to a malicious node."""
        print(f'NOW, you are in assign_attack in attack_simulation.py for attack type: {attack_type}')
        if attack_type == "data_poisoning":
            node.perform_data_poisoning = True
        elif attack_type == "sybil_attack":
            node.perform_sybil_attack = True
        elif attack_type == "evasion_attack":
            node.perform_evasion_attack = True
        elif attack_type == "backdoor_attack":
            node.perform_backdoor_attack = True
        elif attack_type == "model_inversion_attack":
            node.perform_model_inversion_attack = True
        elif attack_type == "membership_inference_attack":
            node.perform_membership_inference_attack = True

    def simulate_data_poisoning(self):
        """Simulates data poisoning attack by modifying local training data of malicious nodes."""
        print('NOW, you are in simulate_data_poisoning in attack_simulation.py')
        for node in self.nodes:
            if getattr(node, 'perform_data_poisoning', False):
                node.data = self.generate_malicious_data(node.data)
                print(f"Node {node.node_id} data has been poisoned.")
        print("Data poisoning simulation completed.")

    def generate_malicious_data(self, data):
        """Generates malicious data to poison the training dataset."""
        print('NOW, you are in generate_malicious_data in attack_simulation.py')
        poisoned_data = []
        for inputs, labels in data:
            poisoned_labels = labels.clone()
            poisoned_labels = (poisoned_labels + 1) % tokenizer.vocab_size
            poisoned_data.append((inputs, poisoned_labels))
        return poisoned_data

    def simulate_sybil_attack(self):
        """Sybil attack creates duplicate nodes."""
        print('NOW, you are in simulate_sybil_attack in attack_simulation.py')
        for node in self.nodes:
            if getattr(node, 'perform_sybil_attack', False):
                for _ in range(5):
                    new_node = copy.deepcopy(node)
                    new_node.node_id = self.generate_new_node_id()
                    self.nodes.append(new_node)
                    print(f"Sybil node {new_node.node_id} created.")

    def simulate_evasion_attack(self):
        """Simulates evasion attack (malicious nodes craft updates to evade detection)."""
        print('NOW, you are in simulate_evasion_attack in attack_simulation.py')
        for node in self.nodes:
            if getattr(node, 'perform_evasion_attack', False):
                print(f"Node {node.node_id} is attempting an evasion attack by crafting deceptive updates.")

    def simulate_backdoor_attack(self):
        """Simulates backdoor attack by injecting triggers into the model updates."""
        print('NOW, you are in simulate_backdoor_attack in attack_simulation.py')
        for node in self.nodes:
            if getattr(node, 'perform_backdoor_attack', False):
                print(f"Node {node.node_id} is attempting a backdoor attack by injecting triggers.")

    def simulate_model_inversion_attack(self):
        """Simulates model inversion attack to reconstruct training data from model outputs."""
        print('NOW, you are in simulate_model_inversion_attack in attack_simulation.py')
        for node in self.nodes:
            if getattr(node, 'perform_model_inversion_attack', False):
                print(f"Node {node.node_id} is attempting a model inversion attack.")

    def simulate_membership_inference_attack(self):
        """Simulates membership inference attack to determine if a data point was part of the training dataset."""
        print('NOW, you are in simulate_membership_inference_attack in attack_simulation.py')
        for node in self.nodes:
            if getattr(node, 'perform_membership_inference_attack', False):
                print(f"Node {node.node_id} is attempting a membership inference attack.")

    def sanitize_data(self):
        """Apply data sanitization methods to all nodes' data."""
        print('NOW, you are in sanitize_data in attack_simulation.py')
        for node in self.nodes:
            node.data = self.remove_duplicates(self.detect_outliers(self.sanitize_node_data(node.data)))
            print(f"Data for Node {node.node_id} has been sanitized.")

    def sanitize_node_data(self, data):
        """Sanitizes node's data by validating inputs and labels."""
        print('NOW, you are in sanitize_node_data in attack_simulation.py')
        sanitized_data = []
        for inputs, labels in data:
            if self.is_valid_input(inputs) and self.is_valid_label(labels):
                sanitized_data.append((inputs, labels))
        return sanitized_data

    def is_valid_input(self, inputs):
        """Checks if inputs are valid."""
        print('NOW, you are in is_valid_input in attack_simulation.py')
        return torch.all(inputs < tokenizer.vocab_size)

    def is_valid_label(self, labels):
        """Check if labels are valid."""
        print('NOW, you are in is_valid_label in attack_simulation.py')
        return torch.all(labels >= 0) and torch.all(labels < tokenizer.vocab_size)

    def detect_outliers(self, data):
        """Detects and removes outliers from the data based on input length."""
        print('NOW, you are in detect_outliers in attack_simulation.py')
        lengths = [len(inputs) for inputs, _ in data]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        threshold = mean_length + 3 * std_length
        filtered_data = [(inputs, labels) for inputs, labels in data if len(inputs) <= threshold]
        return filtered_data

    def remove_duplicates(self, data):
        """Removes duplicate entries from the data."""
        print('NOW, you are in remove_duplicates in attack_simulation.py')
        seen = set()
        unique_data = []
        for inputs, labels in data:
            data_hash = hash((tuple(inputs.tolist()), tuple(labels.tolist())))
            if data_hash not in seen:
                seen.add(data_hash)
                unique_data.append((inputs, labels))
        return unique_data

    def preprocess_text(self, text):
        """Preprocesses text (unpersonalization is important here for any tasks)."""
        print('NOW, you are in preprocess_text in attack_simulation.py')
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def preprocess_dataset(self, dataset):
        """Apply preprocessing to the dataset."""
        print('NOW, you are in preprocess_dataset in attack_simulation.py')
        preprocessed_data = []
        for example in dataset:
            text = self.preprocess_text(example['text'])
            tokens = tokenizer.encode(text, add_special_tokens=True)
            inputs = torch.tensor(tokens[:-1])
            labels = torch.tensor(tokens[1:])
            preprocessed_data.append((inputs, labels))
        return preprocessed_data

    def adversarial_training(self, model, data_loader, optimizer, loss_fn):
        """Trains the model using adversarial examples."""
        print('NOW, you are in adversarial_training in attack_simulation.py')
        for inputs, labels in data_loader:
            inputs_adv = self.generate_adversarial_examples(inputs, labels, model)  # Generate adversarial examples
            combined_inputs = torch.cat([inputs, inputs_adv])  # Combine original and adversarial inputs
            combined_labels = torch.cat([labels, labels])
            optimizer.zero_grad()
            outputs = model(combined_inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), combined_labels.view(-1))
            loss.backward()
            optimizer.step()

    def generate_adversarial_examples(self, inputs, labels, model):
        """Generates adversarial examples."""
        print('NOW, you are in generate_adversarial_examples in attack_simulation.py')
        inputs_adv = inputs.clone().detach()
        inputs_adv.requires_grad = True

        outputs = model(inputs_adv)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        model.zero_grad()
        loss.backward()

        epsilon = 0.01
        perturbation = epsilon * inputs_adv.grad.sign()  # Apply perturbation
        inputs_adv = inputs_adv + perturbation
        inputs_adv = inputs_adv.detach()
        return inputs_adv

    def train_with_privacy(self, model, data_loader, optimizer):
        """Train the model with differential privacy."""
        print('NOW, you are in train_with_privacy in attack_simulation.py')

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
