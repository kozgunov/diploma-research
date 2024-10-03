import torch
import copy
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

class Aggregator:
    def __init__(self, nodes):
        self.nodes = nodes
        self.encrypted_updates = {}
        self.public_keys = [node.public_key for node in nodes]
        self.global_model_state = None  # parameters of global model. none is the beginning
        self.model = copy.deepcopy(nodes[0].model)  # reference model structure

    async def receive_updates(self, node_id, encrypted_updates):
        self.encrypted_updates[node_id] = encrypted_updates
        if len(self.encrypted_updates) == len(self.nodes):
            await self.aggregate_updates()

    async def aggregate_updates(self):
        aggregated_updates = {}
        num_nodes = len(self.nodes)
        for name in self.encrypted_updates[self.nodes[0].node_id]:
            encrypted_sum = None
            for node_id in self.encrypted_updates:
                encrypted_update = self.encrypted_updates[node_id][name]
                if encrypted_sum is None:
                    encrypted_sum = encrypted_update
                else:
                    encrypted_sum = [x + y for x, y in zip(encrypted_sum, encrypted_update)]
            aggregated_updates[name] = encrypted_sum

        decrypted_updates = self.decrypt_updates(aggregated_updates) # decrypt the aggregated updates (requires private key)
        self.update_global_model(decrypted_updates) # update the global model
        for node in self.nodes: # deploy the updated model
            node.set_global_model(self.global_model_state)

        self.encrypted_updates.clear()  # think up the caching and memory design in the future!!!!!!!!!!!!!!!!!!!!!!!!!

    def encrypt_updates(self, delta_w): # homomorphic encrypting
        HE = Pyfhel()
        HE.contextGen(p=65537)
        HE.keyGen()
        encrypted_updates = {}
        for name, param in delta_w.items():
            flattened = param.view(-1).tolist()
            encrypted = [HE.encryptFrac(x) for x in flattened]
            encrypted_updates[name] = encrypted
        self.HE = HE  # store encrypting instances for decryption
        return encrypted_updates

    def decrypt_updates(self, aggregated_updates):
        decrypted_updates = {}
        for name, encrypted_params in aggregated_updates.items():
            # decrypted = [self.nodes[0].private_key.decrypt(x) / len(self.nodes) for x in encrypted_params]
            decrypted = [self.nodes[0].HE.decryptFrac(x) for x in encrypted_params]
            param_shape = self.model.state_dict()[name].shape
            decrypted_param = torch.tensor(decrypted).view(param_shape)
            decrypted_updates[name] = decrypted_param
        return decrypted_updates

    def update_global_model(self, aggregated_updates):
        global_state = self.model.state_dict() # update the global model weight
        for name in aggregated_updates:
            global_state[name] += aggregated_updates[name]
        self.global_model_state = global_state
        self.model.load_state_dict(global_state)
