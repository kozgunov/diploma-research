import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from blake3 import blake3
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

print('NOW, you are in node.py')

def compute_vdf(input_value, difficulty=100000):
    print('NOW, you are in compute_vdf in node.py')
    result = input_value
    modulus = 1000000007 
    for _ in range(difficulty):
        result = pow(result, 2, modulus)
    return result

def validate_proof_of_work(block):
    print('NOW, you are in validate_proof_of_work in node.py')
    block_header = f"{block['creator']}{block['vdf_output']}{block['previous_hash']}"
    computed_hash = blake3(f"{block_header}{block['nonce']}".encode()).hexdigest()
    difficulty = 4
    target = '0' * difficulty
    return computed_hash.startswith(target)

def apply_differential_privacy(delta_w):
    print('NOW, you are in apply_differential_privacy in node.py')
    epsilon = 1.0 # budget 
    sensitivity = 1.0 
    for name in delta_w:
        noise = torch.randn(delta_w[name].shape) * (sensitivity / epsilon)
        delta_w[name] += noise
    return delta_w

class Node:
    def __init__(self, node_id, stake, data, model, is_malicious=False):
        self.node_id = node_id
        self.stake = stake
        self.data = data  # Local dataset
        self.model = model  # Local model
        self.is_malicious = is_malicious  
        self.generate_keys()  # generate ECDSA keys
        self.reputation = 0.0 
        self.learning_rate = 1e-5
        self.batch_size = 32
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.blockchain = []  
        self.vdf_output = None  
        self.initial_model_state = None  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.previous_hash = '0' * 64 
        self.nodes = []  

    # Generates ECDSA key pair for digital signatures
    def generate_keys(self):
        print('NOW, you are in generate_keys in node.py')
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    async def local_training(self, epochs=1):
        print('NOW, you are in local_training in node.py')
        self.initialize_model_state()
        self.model.model.train()
        for epoch in range(epochs):
            for inputs, labels in self.data_loader():
                inputs = inputs.to(self.model.device)
                labels = labels.to(self.model.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        delta_w = self.get_model_updates() 
        if self.is_malicious:
            for name in delta_w:
                delta_w[name] = delta_w[name] * 10
        else:
            delta_w = apply_differential_privacy(delta_w)
        return delta_w  # Return updates directly (removed homomorphic encryption)

    def data_loader(self):
        print('NOW, you are in data_loader in node.py')
        dataset = self.data
        indices = list(range(len(dataset)))
        random.shuffle(indices)  # Randomize the data
        for start_idx in range(0, len(dataset), self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            batch = [dataset[i] for i in excerpt]
            inputs, labels = zip(*batch)
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            yield inputs, labels

    def initialize_model_state(self):
        print('NOW, you are in initialize_model_state in node.py')
        self.initial_model_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def get_model_updates(self):
        print('NOW, you are in get_model_updates in node.py')
        delta_w = {}
        for name, param in self.model.named_parameters():
            delta_w[name] = param.data - self.initial_model_state[name]
        return delta_w

    async def send_updates(self, aggregator):
        print('NOW, you are in send_updates in node.py')
        updates = await self.local_training(epochs=1)
        await aggregator.receive_updates(self.node_id, updates)

    def sign_block(self, message):
        message_bytes = message.encode()
        signature = self.private_key.sign(
            message_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        return signature

    async def create_block(self, all_nodes):
        print('NOW, you are in create_block in node.py')
        vdf_input = random.randint(0, int(1e6))
        self.vdf_output = compute_vdf(vdf_input, difficulty=100000)
        difficulty = 4  # Number of leading zeros required
        block_header = f"{self.node_id}{self.vdf_output}{self.previous_hash}"
        nonce, block_hash = self.proof_of_work(block_header, difficulty)
        block = {
            'creator': self.node_id,
            'vdf_output': self.vdf_output,
            'nonce': nonce,  # Do not modify nonce
            'hash': block_hash,
            'transactions': [],  # Placeholder for future transactions
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None,
            'signature': self.sign_block(block_header + str(nonce))
        }
        self.previous_hash = block['hash']  # Update previous hash
        block['hash'] = blake3(str(block).encode()).hexdigest()
        
        for node in all_nodes:
            if node.node_id != self.node_id:
                node.receive_block(block)

        self.blockchain.append(block)  # Add block to own blockchain

    def validate_block(self, block):
        print('NOW, you are in validate_block in node.py')
        if block['previous_hash'] != self.previous_hash:
            return False
        if not self.signature_verification(block):
            return False
        if not validate_proof_of_work(block):
            return False
        return True  # Block is valid if all checks pass

    def receive_block(self, block):
        print('NOW, you are in receive_block in node.py')
        if self.validate_block(block):
            self.blockchain.append(block)
            self.previous_hash = block['hash']  # Update previous hash
        else:
            print(f"Node {self.node_id}: Invalid block received from Node {block['creator']}")

    def set_global_model(self, global_model_state):
        print('NOW, you are in set_global_model in node.py')
        self.model.model.load_state_dict(global_model_state)

    def update_reputations(self, comment):
        print('NOW, you are in update_reputations in node.py')
        MAX_REPUTATION = 1000  # Maximum reputation
        if comment == 'malicious':
            self.reputation = max(0, int(self.reputation - 3))
            print('Node is malicious')
        elif comment == 'honest':
            self.reputation = min(self.reputation + 1, MAX_REPUTATION)
            print('Node is honest')

    def proof_of_work(self, block_header, difficulty):
        print('NOW, you are in proof_of_work in node.py')
        nonce = 0
        target = '0' * difficulty
        while True:
            hash_result = blake3(f"{block_header}{nonce}".encode()).hexdigest()
            if hash_result.startswith(target):
                return nonce, hash_result
            else:
                nonce += 1  # Increment nonce

    def signature_verification(self, block):
        print('NOW, you are in signature_verification in node.py')
        creator_public_key = self.get_public_key(block['creator'])
        if creator_public_key is None:
            return False
        message = (f"{block['creator']}{block['vdf_output']}{block['previous_hash']}{block['nonce']}").encode()
        signature = block['signature']
        try:
            creator_public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            return False

    async def participate_in_consensus(self, all_nodes):
        print('NOW, you are in participate_in_consensus in node.py')
        total_stake_reputation = sum(node.stake * node.reputation for node in all_nodes)
        if total_stake_reputation == 0:
            selection_probability = 1 / len(all_nodes)
        else:
            selection_probability = (self.stake * self.reputation) / total_stake_reputation
        selected_node = random.choices(all_nodes, weights=[node.stake * node.reputation for node in all_nodes], k=1)[0]
        return selected_node

    def get_public_key(self, node_id):
        print('NOW, you are in get_public_key in node.py')
        for node in self.nodes:
            if node.node_id == node_id:
                return node.public_key
        return None
