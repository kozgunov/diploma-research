import copy
import random
import main
import torch

from node import Node, encrypt_updates


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


def malicious_dataset_attack(node):
    node.data = generate_malicious_data()


def generate_malicious_data():
    malicious_data = []
    return malicious_data


def sybil_attack(nodes):
    malicious_node = nodes[0]
    for _ in range(sybil_nodes_count):
        new_node = copy.deepcopy(malicious_node)
        new_node.node_id = generate_new_node_id()
        nodes.append(new_node)

    def evasion_attack(node):
        pass

    def backdoor_attack(node):
        pass


# recursive function...
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

