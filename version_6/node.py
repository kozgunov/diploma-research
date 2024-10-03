from crypto.PublicKey import RSA
from crypto.Random import random as crypto_random
from blake3 import blake3
import random
import torch
import torch.nn as nn
from crypto.Signature import pkcs1_15
from torch.utils.data import DataLoader





def encrypt_updates(delta_w):  # encrypt the model updates
    encrypted_updates = {}
    for name, param in delta_w.items():
        flattened = param.view(-1).tolist()
        encrypted = [x + crypto_random.randint(1, 10) for x in flattened]
        encrypted_updates[name] = encrypted
    return encrypted_updates


def compute_vdf(input_value):
    result = input_value
    for _ in range(100000):  # simulate time delay
        result = pow(result, 2, int(1e9 + 7))
    return result


def validate_block(block):
    return True


def apply_differential_privacy(delta_w):
    epsilon = 1.0  # privacy budget
    sensitivity = 1.0  # depends on the data and model
    for name in delta_w:
        noise = torch.randn(delta_w[name].shape) * (sensitivity / epsilon)
        delta_w[name] += noise
    return delta_w




class Node:
    def __init__(self, node_id, stake, data, model, is_malicious=False):
        self.epochs = 1 # initial number of epochs, implying quite many nodes in the beginning
        self.node_id = node_id
        self.stake = stake
        self.data = data  # local dataset
        self.model = model  # local model (copied of global model afterward)
        self.is_malicious = is_malicious  # flag to simulate malicious behavior
        self.public_key = None
        self.private_key = None
        self.generate_keys()
        self.reputation = 0.0  # starts at 0 and goes up to infinity
        self.learning_rate = 1e-5
        self.batch_size = 32
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.blockchain = []  # local copy of the blockchain
        self.vdf_output = None  # for PoT
        self.initial_model_state = None  # store model state before training
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.previous_hash = '0' * 64 # initial prev.hash

    def generate_keys(self):  # generate RSA key pair for digital signatures
        key = RSA.generate(2048)
        self.public_key = key.publickey()
        self.private_key = key

    async def local_training(self, epochs=1):  # store initial model state
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        delta_w = self.get_model_updates()
        if self.is_malicious:  # check poisoning data attack
            for name in delta_w:
                delta_w[name] = delta_w[name] * 10
        encrypted_updates = encrypt_updates(delta_w)
        return encrypted_updates

    #def data_loader(self):
    #    dataset = self.data
    #    loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    #    for batch in loader:
    #        yield batch

    def data_loader(self):  # simulate loading data
        dataset = self.data
        random.shuffle(self.data) # randomize it
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for start_idx in range(0, len(dataset), self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            batch = [dataset[i] for i in excerpt]
            inputs, labels = zip(*batch)
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            yield inputs, labels

    def initialize_model_state(self):  # store initial model state before training from prev. training
        self.initial_model_state = \
            {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }

    def get_model_updates(self):  # compute the difference between updated model and initial model via the metrics
        delta_w = {}
        for name, param in self.model.named_parameters():
            delta_w[name] = param.data - self.initial_model_state[name]
        return delta_w

    async def send_updates(self, aggregator):
        encrypted_updates = await self.local_training(epochs=self.epochs)
        await aggregator.receive_updates(self.node_id, encrypted_updates)

    async def create_block(self, all_nodes):  # simulate VDF for PoT
        vdf_input = random.randint(0, int(1e6))
        self.vdf_output = compute_vdf(vdf_input)
        nonce = 0
        target = int(1e75)  # adjusted difficulty
        difficulty = 4  # number of leading zeros
        block_header = f"{self.node_id}{self.vdf_output}{self.previous_hash}"
        nonce, block_hash = self.proof_of_work(block_header, difficulty)

        # create block
        block = {
            'creator': self.node_id,
            'vdf_output': self.vdf_output,
            'nonce': nonce,  # NEVER TOUCH IT
            'hash': block_hash,
            'transactions': [],  # for cryptoproducts realization
            #'previous_hash': self.previous_hash,
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None,
        }
        block['hash'] = blake3(str(block).encode()).hexdigest()

        for node in all_nodes:
            if node.node_id != self.node_id:
                node.receive_block(block)

        self.blockchain.append(block)  # add block to own blockchain

    def receive_block(self, block):  # validate the block
        if self.validate_block(block):
            self.blockchain.append(block)
            self.previous_hash = block['hash']  # Update previous hash
        else:
            print(f"Node {self.node_id}: Invalid block received from Node {block['creator']}")

    def set_global_model(self, global_model_state):  # update the local model with global one
        self.model.load_state_dict(global_model_state)

    def update_reputations(self, node, comment): #upgrade/downgrade of reputation for PoR
        MAX_REPUTATION = int(1000) # the biggest reputation
        DEFLATION = int(0.99 * self.reputation) # we wanna decrease the inflation, which will appear in n rounds (for the future)
        if comment == 'malicious':
            self.reputation = max(0, int(self.reputation - 3))
            print('node is malicious')
        elif comment == 'honest':
            self.reputation = min(self.reputation + 1, MAX_REPUTATION)
            print('node is honest')

    def normalize_reputations(self, nodes): # we don't use this function, if we wanna to shorten the distance between their reputations (as for me, we can folllow the integers)
        total_reputation = sum(node.reputation for node in nodes)
        for node in nodes:
            node.reputation = (node.reputation / total_reputation)

    def proof_of_work(self, block_header, difficulty):
        nonce = 0
        target = '0' * difficulty
        while True:
            hash_result = blake3(f"{block_header}{nonce}".encode()).hexdigest()
            if hash_result[:difficulty] == target:
                return nonce, hash_result
            else:
                nonce += 1

    def validate_block(self, block):  # validate previous hash
        if block['previous_hash'] != self.previous_hash:
            return False
        if not self.signature_verification(block):  # validate signature
            return False
        if not self.validate_proof_of_work(block):  # validate Proof-of-Work
            return False
        return True

    def signature_verification(self, block):
        creator_public_key = self.get_public_key(block['creator'])
        message = str(block['data']).encode()   # prepare the message
        h = blake3.new(message) # encrypt this massage
        try:
            pkcs1_15.new(creator_public_key).verify(h, block['signature'])
            return True
        except (ValueError, TypeError):
            return False

    def validate_proof_of_work(self, block):
        block_header = f"{block['creator']}{block['vdf_output']}{block['previous_hash']}"
        computed_hash = blake3(f"{block_header}{block['nonce']}".encode()).hexdigest()
        difficulty = 4
        target = '0' * difficulty
        return computed_hash.startswith(target)

    async def participate_in_consensus(self, all_nodes):  # PoR selection
        total_stake_reputation = sum(node.stake * node.reputation for node in all_nodes)
        if total_stake_reputation == 0:
            selection_probability = 1 / len(all_nodes)  # exception handling
        else:
            selection_probability = [(self.stake * self.reputation) / total_stake_reputation for node in all_nodes]
        selected_node = random.choices(all_nodes, weights=selection_probability, k=1)[0]
        return selected_node

#if random.random() < selection_probability:
#    await self.create_block(all_nodes)
#else:
#    pass
