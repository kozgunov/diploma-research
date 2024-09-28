import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Process, Queue
from asyncio import Queue as AsyncQueue
import asyncio
import random
from crypto.paillier import PaillierPublicKey, PaillierPrivateKey, generate_paillier_keypair
from hashlib import sha256
import time

class Node:
    def __init__(self, node_id, stake, data, model, is_malicious=False):
        self.node_id = node_id
        self.stake = stake
        self.data = data  # local splitted dataset
        self.model = model  # local model (copy of global model)
        self.is_malicious = is_malicious  # flag to simulate malicious behavior
        self.public_key = None
        self.private_key = None
        self.generate_keys()
        self.reputation = 1.0  # starting reputation score for PoR/PoS
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.incoming_updates = AsyncQueue() # asynchronous queues for communication
        self.incoming_blocks = AsyncQueue()
        self.blockchain = []
        self.lingua_coins = 100  # starting balance
        self.vdf_output = None  # for PoT

    def generate_keys(self):
        self.public_key, self.private_key = generate_paillier_keypair()

    async def local_training(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in self.data_loader():
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
        delta_w = self.get_model_updates()
        encrypted_updates = self.encrypt_updates(delta_w)
        return encrypted_updates

    def data_loader(self): # simulate a data loader
        dataset = self.data
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for start_idx in range(0, len(dataset), self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            inputs, labels = zip(*[dataset[i] for i in excerpt])
            inputs = torch.stack(inputs)
            labels = torch.tensor(labels)
            yield inputs, labels

    def get_model_updates(self): # compute difference between updated model and initial model
        delta_w = {}
        for name, param in self.model.named_parameters():
            delta_w[name] = param.data - self.initial_model_state[name]
        return delta_w

    def encrypt_updates(self, delta_w): # encrypt model updates using encryption
        encrypted_updates = {}
        for name, param in delta_w.items():
            flattened = param.view(-1).tolist()
            encrypted = [self.public_key.encrypt(x) for x in flattened]
            encrypted_updates[name] = encrypted
        return encrypted_updates

    async def participate_in_consensus(self): # Implement PoS, PoT, and minimal PoW
        pass

    async def send_updates(self, aggregator):
        encrypted_updates = await self.local_training()
        await aggregator.receive_updates(self.node_id, encrypted_updates)

    def receive_block(self, block): # add block to local blockchain copy
        self.blockchain.append(block)

    def set_global_model(self, global_model_state): # update local model with new global model state
        self.model.load_state_dict(global_model_state)

    def initialize_model_state(self): # store initial model before training
        self.initial_model_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    async def local_training(self, epochs=1):
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
    if self.is_malicious:
        for name in delta_w:
            delta_w[name] = delta_w[name] * 10 
    encrypted_updates = self.encrypt_updates(delta_w)
    return encrypted_updates

    # Additional methods for VDF computation, signature verification, etc.

