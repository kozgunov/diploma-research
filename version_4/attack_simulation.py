import copy

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
