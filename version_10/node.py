import random
import torch
import torch.nn as nn
from blake3 import blake3
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature
from consensus_mechanism import compute_vdf, proof_of_work, integrate_consensus_mechanisms
from blockchain_structure import Blockchain, Block
from attack_simulation import apply_differential_privacy
import time
import psutil

print('NOW, you are in node.py')

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
        self.blockchain = Blockchain()  # Use Blockchain class instead of list
        self.vdf_output = None  
        self.initial_model_state = None  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.previous_hash = '0' * 64 
        self.nodes = []  
        self.epochs = 1  # Add default epochs parameter
        self.consensus_participations = 0
        self.successful_validations = 0
        
    # Generates ECDSA key pair for digital signatures
    def generate_keys(self):
        print('NOW, you are in generate_keys in node.py')
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    async def local_training(self, epochs=1):
        """Performs local training on the node's data."""
        print(f"\n=== Starting local training for Node {self.node_id} ===")
        print(f"Initial model state hash: {hash(str(self.model.state_dict()))}")
        
        try:
            self.initialize_model_state()
            self.model.model.train()
            print("Model training started.")
            
            total_loss = 0  # Initialize total_loss
            batch_count = 0  # Initialize batch_count
            
            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")
                
                for inputs, labels in self.data_loader():
                    try:
                        # Move data to device
                        inputs = inputs.to(self.model.device)
                        labels = labels.to(self.model.device)
                        
                        # Training step
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                        
                        print(f"Batch {batch_count+1} - Loss: {loss.item():.4f}")
                        
                        # Backward pass
                        loss.backward()
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                        batch_count += 1
                        
                    except Exception as e:
                        print(f"ERROR in batch processing: {str(e)}")
                        continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
            
            # Get model updates
            delta_w = self.get_model_updates()  # Ensure delta_w is defined here
            print("Model updates computed successfully.")
            
            # Apply security measures
            if self.is_malicious:
                print("WARNING: Applying malicious modifications")
                for name in delta_w:
                    delta_w[name] = delta_w[name] * 10
            else:
                print("Applying differential privacy")
                delta_w = apply_differential_privacy(delta_w)
                
            print(f"Final model state hash: {hash(str(self.model.state_dict()))}")
            return delta_w  # Return delta_w after it has been defined
            
        except Exception as e:
            print(f"CRITICAL ERROR in local training: {str(e)}")
            raise

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
        self.vdf_output = compute_vdf(vdf_input)
        
        # Simplified PoW
        block_header = f"{self.node_id}{self.vdf_output}{self.previous_hash}"
        nonce, block_hash = self.simple_proof_of_work(block_header)  # Use simplified PoW
        
        # Create proper Block object
        block = Block(
            index=len(self.blockchain.chain),
            previous_hash=self.previous_hash,
            timestamp=time.time(),
            data={'vdf_output': self.vdf_output, 'transactions': []},
            creator=self.node_id,
            nonce=nonce,
            hash_value=block_hash
        )
        
        # Broadcast block
        for node in all_nodes:
            if node.node_id != self.node_id:
                await node.receive_block(block)

    def simple_proof_of_work(self, block_header):
        """Simplified Proof of Work to reduce computational burden."""
        print('NOW, you are in simple_proof_of_work in node.py')
        nonce = 0
        # Lower difficulty for faster block creation
        target = '0' * 2  # Example: lower difficulty
        while True:
            hash_result = blake3(f"{block_header}{nonce}".encode()).hexdigest()
            if hash_result.startswith(target):
                return nonce, hash_result
            else:
                nonce += 1  # Increment nonce

    def validate_block(self, block):
        print('NOW, you are in validate_block in node.py')
        if block['previous_hash'] != self.previous_hash:
            return False
        if not self.signature_verification(block):
            return False
        if not self.blockchain.validate_proof_of_work(block):
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
        """Participate in consensus using all three mechanisms"""
        if integrate_consensus_mechanisms(self, all_nodes):
            # Validate model updates before creating a block
            delta_w = await self.local_training()
            if self.validate_model_update(delta_w):
                await self.create_block(all_nodes)
            else:
                print(f"Node {self.node_id} rejected invalid model update.")

    def get_public_key(self, node_id):
        print('NOW, you are in get_public_key in node.py')
        for node in self.nodes:
            if node.node_id == node_id:
                return node.public_key
        return None

    async def get_training_loss(self):
        """Calculate current training loss"""
        total_loss = 0
        num_batches = 0
        
        for inputs, labels in self.data_loader():
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def validate_model_update(self, delta_w):
        """Validate model updates before applying them."""
        print('NOW, you are in validate_model_update in node.py')
        # Implement validation logic, e.g., check for NaN values
        for name, update in delta_w.items():
            if torch.isnan(update).any():
                print(f"Invalid update detected for {name}. Aborting update.")
                return False
        print("Model update validated successfully.")
        return True

    def can_load_model(self):
        """Check if the node can load the LLaMA 2.7B model based on its resources."""
        print(f"Checking if Node {self.node_id} can load the model...")
        model_memory_size = 2.7 * 4 / (1024 ** 2)  # in GB
        additional_memory_factor = 2  # Estimate for activations and other overhead
        total_memory_required = model_memory_size * additional_memory_factor
        
        memory = psutil.virtual_memory()
        if memory.available < total_memory_required * (1024 ** 2):  # Convert GB to bytes
            print(f"Node {self.node_id} cannot load the model: Insufficient memory.")
            return False
        print(f"Node {self.node_id} can load the model.")
        return True
