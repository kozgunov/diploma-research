from crypto.PublicKey import RSA
from crypto.Random import random as crypto_random
from hashlib import sha256
import asyncio
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time

class Node:
    def __init__(self, node_id, stake, data, model, is_malicious=False):
        self.node_id = node_id
        self.stake = stake
        self.data = data  # local dataset
        self.model = model  # local model (copied of global model afterward)
        self.is_malicious = is_malicious  # flag to simulate malicious behavior
        self.public_key = None
        self.private_key = None
        self.generate_keys()
        self.reputation = 1.0  # starting reputation score
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.blockchain = []  # local copy of the blockchain
        self.vdf_output = None  # for PoT
        self.initial_model_state = None  # store model state before training

    def generate_keys(self): # generate RSA key pair for digital signatures
        key = RSA.generate(2048)
        self.public_key = key.publickey()
        self.private_key = key

    async def local_training(self, epochs=5): # store initial model state
        number_of_batches = len(self.data) // self.batch_size
        self.initialize_model_state()
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in self.data_loader():
                inputs = inputs.to(self.model.device)
                labels = labels.to(self.model.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                self.optimizer.step()
        delta_w = self.get_model_updates()
        if self.is_malicious: # check poisoning data attack
            for name in delta_w:
                delta_w[name] = delta_w[name] * 10
        encrypted_updates = self.encrypt_updates(delta_w)
        return encrypted_updates

    def data_loader(self): # simulate loading data
        dataset = self.data
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for start_idx in range(0, len(dataset), self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            batch = [dataset[i] for i in excerpt]
            inputs, labels = zip(*batch)
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            yield inputs, labels

    def initialize_model_state(self): # store initial model state before training from prev. training
        self.initial_model_state = \
        {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def get_model_updates(self): # compute the difference between updated model and initial model via the metrics
        delta_w = {}
        for name, param in self.model.named_parameters():
            delta_w[name] = param.data - self.initial_model_state[name]
        return delta_w

    def encrypt_updates(self, delta_w): # encrypt the model updates
        encrypted_updates = {}
        for name, param in delta_w.items():
            flattened = param.view(-1).tolist()
            encrypted = [x + crypto_random.randint(1, 10) for x in flattened]
            encrypted_updates[name] = encrypted
        return encrypted_updates

    async def send_updates(self, aggregator):
        encrypted_updates = await self.local_training()
        await aggregator.receive_updates(self.node_id, encrypted_updates)

    async def participate_in_consensus(self, all_nodes): # PoS selection without reputation now
        total_stake_reputation = sum(node.stake * node.reputation for node in all_nodes)
        selection_probability = (self.stake * self.reputation) / total_stake_reputation
        if random.random() < selection_probability:
            await self.create_block(all_nodes) # selected block
        else:
            pass

    async def create_block(self, all_nodes): # simulate VDF for PoT
        vdf_input = random.randint(0, int(1e6))
        self.vdf_output = self.compute_vdf(vdf_input)
        nonce = 0
        target = int(1e75)  # adjusted difficulty
        block_header = f"{self.node_id}{self.vdf_output}"
        while int(sha256(f"{block_header}{nonce}".encode()).hexdigest(), 16) > target:
            nonce += 1

        # create block
        block = {
            'creator': self.node_id,
            'vdf_output': self.vdf_output,
            'nonce': nonce, # NEVER TOUCH IT
            'transactions': [],  # for cryptoproducts realization
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None,
        }
        block['hash'] = sha256(str(block).encode()).hexdigest()

        for node in all_nodes:
            if node.node_id != self.node_id:
                node.receive_block(block)

        self.blockchain.append(block) # add block to own blockchain

    def compute_vdf(self, input_value):
        result = input_value
        for _ in range(100000):  # simulate time delay
            result = pow(result, 2, int(1e9 + 7))
        return result

    def receive_block(self, block): # validate the block
        if self.validate_block(block):
            self.blockchain.append(block)
        else:
            print(f"Node {self.node_id}: Invalid block received from Node {block['creator']}")

    def validate_block(self, block):
        return True

    def set_global_model(self, global_model_state): # update the local model with global one
        self.model.load_state_dict(global_model_state)

    def signature_verification(self, block):
        pass

    def apply_differential_privacy(self, delta_w):
        epsilon = 1.0  # privacy budget
        sensitivity = 1.0  # depends on the data and model
        for name in delta_w:
            noise = torch.randn(delta_w[name].shape) * (sensitivity / epsilon)
            delta_w[name] += noise
        return delta_w
