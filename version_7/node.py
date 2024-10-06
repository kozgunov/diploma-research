from crypto.Random import random as crypto_random
from blake3 import blake3
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from crypto.PublicKey import RSA
from crypto.Signature import pkcs1_15
from crypto.Hash import SHA256
from Pyfhel import Pyfhel

print('NOW, you are in node.py')

# less randomization in order to stabilize training of the model

def compute_vdf(self, input_value): # computation of VDFs itself
    print('NOW, you are in compute_vdf in node.py')
    result = input_value
    difficulty = 100000 # might be done as variable of function
    modulus = 1000000007  # Large prime number or int(1e9 + 7)
    for _ in range(difficulty):  # simulate time delay
        result = pow(result, 2, modulus)
    return result


def validate_proof_of_work(block): # obvious function
    print('NOW, you are in validate_proof_of_work in node.py')
    block_header = f"{block['creator']}{block['vdf_output']}{block['previous_hash']}"
    computed_hash = blake3(f"{block_header}{block['nonce']}".encode()).hexdigest()
    difficulty = 4
    target = '0' * difficulty
    return computed_hash.startswith(target)


class Node:
    def __init__(self, node_id, stake, data, model, is_malicious=False):
        self.nodes = []
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


    def set_encryption_parameters(self, context_bytes, public_key_bytes): # homomorphic encryption
        print('NOW, you are in set_encryption_parameters in node.py')
        self.HE = Pyfhel()
        self.HE.from_bytes_context(context_bytes) # serialized encryption context
        self.HE.from_bytes_public_key(public_key_bytes) # serialized public key

    def generate_keys(self):  # generate RSA key pair for digital signatures
        print('NOW, you are in generate_keys in node.py')
        key = RSA.generate(2048)
        self.public_key = key.publickey()
        self.private_key = key

    async def local_training(self, epochs=1):  # local training on node's data
        print('NOW, you are in local_training in node.py')
        number_of_batches = len(self.data) // self.batch_size
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
        delta_w = self.get_model_updates() # get the model's updates
        if self.is_malicious:  # check poisoning data attack
            for name in delta_w:
                delta_w[name] = delta_w[name] * 10 # exaggerate the updates
        encrypted_updates = self.encrypt_updates(delta_w) # encrypt the updates via homomorphic encryption
        return encrypted_updates


    def data_loader(self):  # create data loader for the node's local data
        print('NOW, you are in data_loader in node.py')
        dataset = self.data
        indices = list(range(len(dataset)))
        random.shuffle(indices)# randomize it
        for start_idx in range(0, len(dataset), self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            batch = [dataset[i] for i in excerpt]
            inputs, labels = zip(*batch)
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            yield inputs, labels


    def initialize_model_state(self):  # store initial model state before training from prev. training
        print('NOW, you are in initialize_model_state in node.py')
        self.initial_model_state = \
            {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }

    def get_model_updates(self):  # compute the difference between updated model and initial model via the metrics
        print('NOW, you are in get_model_updates in node.py')
        delta_w = {}
        for name, param in self.model.named_parameters():
            delta_w[name] = param.data - self.initial_model_state[name]
        return delta_w

    async def send_updates(self, aggregator): # sends encrypted updates to aggregator
        print('NOW, you are in send_updates in node.py')
        encrypted_updates = await self.local_training(epochs=self.epochs)
        await aggregator.receive_updates(self.node_id, encrypted_updates)

    def sign_block(self, message): # sign a block with application of private key of its node
        h = SHA256.new(message.encode()) # find out alternative signature with optimized signatures' algorithms
        signature = pkcs1_15.new(self.private_key).sign(h)
        return signature

    async def create_block(self, all_nodes):  # simulate VDF for PoT
        print('NOW, you are in create_block in node.py')
        vdf_input = random.randint(0, int(1e6))
        self.vdf_output = compute_vdf(vdf_input) # compute VDF output for PoT
        nonce = 0 # exception handling
        target = int(1e75)  # adjusted difficulty
        difficulty = 4  # number of leading zeros for finding valid nonce
        block_header = f"{self.node_id}{self.vdf_output}{self.previous_hash}"
        nonce, block_hash = self.proof_of_work(block_header, difficulty)

        # create block/broadcast
        block = \
        {
            'creator': self.node_id,
            'vdf_output': self.vdf_output,
            'nonce': nonce,  # NEVER TOUCH IT
            'hash': block_hash,
            'transactions': [],  # for cryptoproducts realization & future transactions
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None, # or input 'previous_hash': self.previous_hash,
            'signature': self.sign_block(block_header + str(nonce))
        }
        self.previous_hash = block['hash'] # for the case "or" in broadcast of block
        block['hash'] = blake3(str(block).encode()).hexdigest()

        for node in all_nodes:
            if node.node_id != self.node_id:
                node.receive_block(block) # add the block to the local blockchain (might be eliminated in practice)

        self.blockchain.append(block)  # add block to own blockchain

    def validate_block(self, block):  # validate a received block.
        print('NOW, you are in validate_block in node.py')
        if block['previous_hash'] != self.previous_hash:  # start with validation of previous hash
            return False
        if not self.signature_verification(block):  # validate signature
            return False
        if not validate_proof_of_work(block):  # validate Proof-of-Work
            return False
        return True  # 3-of-3 means correct block by definition(or extremely good luck)

    def receive_block(self, block):  # receive and validate a block from another node
        print('NOW, you are in receive_block in node.py')
        if self.validate_block(block):
            self.blockchain.append(block)
            self.previous_hash = block['hash']  # Update previous hash
        else:
            print(f"Node {self.node_id}: Invalid block received from Node {block['creator']}")

    def set_global_model(self, global_model_state):  # update the local model with global one
        print('NOW, you are in set_global_model in node.py')
        self.model.model.load_state_dict(global_model_state)

    def update_reputations(self, comment): #upgrade/downgrade of reputation for PoR
        print('NOW, you are in update_reputations in node.py')
        MAX_REPUTATION = int(1000) # the biggest reputation
        DEFLATION = int(0.99 * self.reputation) # we wanna decrease the inflation, which will appear in n rounds (for the future)
        if comment == 'malicious':
            self.reputation = max(0, int(self.reputation - 3)) # exception handling also here)
            print('node is malicious')
        elif comment == 'honest':
            self.reputation = min(self.reputation + 1, MAX_REPUTATION)
            print('node is honest')


    def proof_of_work(self, block_header, difficulty): # PoW for finding nonce
        print('NOW, you are in proof_of_work in node.py')
        nonce = 0
        target = '0' * difficulty
        while True:
            hash_result = blake3(f"{block_header}{nonce}".encode()).hexdigest()
            # if hash_result[:difficulty] == target: - we can use it, but optimize with computations
            if hash_result.startswith(target):
                return nonce, hash_result
            else:
                nonce += 1

    def signature_verification(self, block): # verify the signature
        print('NOW, you are in signature_verification in node.py')
        creator_public_key = self.get_public_key(block['creator']) # retrieve the public key of the block's creator
        if creator_public_key is None:
            return False
        message = str(block['data']).encode()   # prepare the message for signing
        #message = f"{block['creator']}{block['vdf_output']}{block['previous_hash']}{block['nonce']}" # prepare the message for signing
        h = SHA256.new(message) # encrypt this message -- optimize this function with faster algorithms
        try:
            pkcs1_15.new(creator_public_key).verify(h, block['signature'])
            return True
        except (ValueError, TypeError):
            return False

    async def participate_in_consensus(self, all_nodes):  # PoR selection (taking part in consensuses)
        print('NOW, you are in participate_in_consensus in node.py')
        total_stake_reputation = sum(node.stake * node.reputation for node in all_nodes)
        if total_stake_reputation == 0:
            selection_probability = 1 / len(all_nodes)  # exception handling
        else:
            selection_probability = (self.stake * self.reputation) / total_stake_reputation
        selected_node = random.choices(all_nodes, weights=selection_probability, k=1)[0] # weighted voting
        return selected_node
        #if random.random() < selection_probability: # aggressive mode (decide if create block based on the selection probability)
        #    await self.create_block(all_nodes)


    def encrypt_updates(self, delta_w):  # encrypt the model updates
        print('NOW, you are in encrypt_updates in node.py')
        encrypted_updates = {}
        for name, param in delta_w.items():
            flattened = param.view(-1).tolist()
            # encrypted = [x + crypto_random.randint(1, 10) for x in flattened] - alternative, more flexible (if use this way, then write out of the class)
            encrypted = [self.HE.encryptFrac(x) for x in flattened]
            encrypted_updates[name] = encrypted
        return encrypted_updates

    def get_public_key(self, node_id): # get the public key of a node given its ID
        print('NOW, you are in get_public_key in node.py')
        for node in self.nodes:
            if node.node_id == node_id:
                return node.public_key
        return None



    # we can also use the following functions:


    def normalize_reputations(self, nodes): # we don't use this function, if we wanna to shorten the distance between their reputations (as for me, we can folllow the integers)
        print('NOW, you are in normalize_reputations in node.py')
        total_reputation = sum(node.reputation for node in nodes)
        for node in nodes:
            node.reputation = (node.reputation / total_reputation)



def apply_differential_privacy(delta_w):
    print('NOW, you are in apply_differential_privacy in node.py')
    epsilon = 1.0  # privacy budget
    sensitivity = 1.0  # depends on the data and model
    for name in delta_w:
        noise = torch.randn(delta_w[name].shape) * (sensitivity / epsilon)
        delta_w[name] += noise
    return delta_w
