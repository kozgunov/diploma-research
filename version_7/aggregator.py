import torch
import copy
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import asyncio

print('NOW, you are in aggregator.py')


class Aggregator:
    def __init__(self, nodes):
        self.nodes = nodes
        self.encrypted_updates = {}
        self.public_keys = [node.public_key for node in nodes]
        self.global_model_state = None  # parameters of global model. none is in the beginning
        self.model = copy.deepcopy(
            nodes[0].model.model)  # reference model structure & copying the model structure from the best of the nodes
        self.HE = Pyfhel()  # initialize Pyfhel for homomorphic encryption
        self.HE.contextGen(p=65537)  # generate context with prime modulus p (typically written with that's number)
        self.HE.keyGen()  # generate public and private keys
        context_bytes = self.HE.to_bytes_context()  # context2nodes
        public_key_bytes = self.HE.to_bytes_public_key()  # publickey2nodes
        for node in nodes:
            node.set_encryption_parameters(context_bytes, public_key_bytes)
        self.aggregation_event = asyncio.Event()  # event to completion of aggregation


    async def receive_updates(self, node_id, encrypted_updates):
        print('NOW, you are in receive_updates in aggregator.py')
        self.encrypted_updates[node_id] = encrypted_updates
        if len(self.encrypted_updates) == len(self.nodes):  # if updates from all nodes received -> continue aggregation
            await self.aggregate_updates()

    async def aggregate_updates(self):  # aggregate encrypted updates from all nodes using homomorphic encryption.
        print('NOW, you are in aggregate_updates in aggregator.py')
        aggregated_updates = {}
        num_nodes = len(self.nodes)
        param_names = self.encrypted_updates[next(iter(self.encrypted_updates))].keys()  # list with parameters' names
        for name in param_names:
            encrypted_sum = None
            for node_id in self.encrypted_updates:  # sum the encrypted updates for each parameter
                encrypted_update = self.encrypted_updates[node_id][name]
                if encrypted_sum is None:
                    encrypted_sum = encrypted_update
                else:
                    encrypted_sum = [x + y for x, y in
                                     zip(encrypted_sum, encrypted_update)]  # homomorphic addition of ciphertexts
            aggregated_updates[name] = encrypted_sum

        decrypted_updates = self.decrypt_updates(
            aggregated_updates)  # decrypt the aggregated updates (requires private key)
        self.update_global_model(decrypted_updates)  # update the global model by decrypted updates
        for node in self.nodes:  # deploy the updated model for all submodels
            node.set_global_model(self.global_model_state)

        self.encrypted_updates.clear()  # think up the caching and memory design in the future!!!!!!!!!!!!!!!!!!!!!!!!!
        self.aggregation_event.set()  # set the aggregation event

    def decrypt_updates(self, aggregated_updates):  # decrypt them using private keys
        print('NOW, you are in decrypt_updates in aggregator.py')
        decrypted_updates = {}
        for name, encrypted_params in aggregated_updates.items():
            # decrypted = [self.nodes[0].private_key.decrypt(x) / len(self.nodes) for x in encrypted_params]
            decrypted = [self.nodes[0].HE.decryptFrac(x) for x in encrypted_params]  # decrypt each parameter
            param_shape = self.model.state_dict()[name].shape
            decrypted_param = torch.tensor(decrypted).view(param_shape)
            decrypted_updates[name] = decrypted_param
        return decrypted_updates

    def update_global_model(self, aggregated_updates):
        print('NOW, you are in update_global_model in aggregator.py')
        global_state = self.model.state_dict()  # update the global model weight
        for name in aggregated_updates:  # update all parameters
            global_state[name] += aggregated_updates[name] / len(self.nodes)
        self.model.load_state_dict(global_state)  # load the updated model
        self.global_model_state = global_state  # update model

    async def wait_for_aggregation(self):  # until it will be finished...
        print('NOW, you are in wait_for_aggregation in aggregator.py')
        await self.aggregation_event.wait()
        self.aggregation_event.clear()

    '''
    this function integrated into __init__ and might be removed HERE or THERE (into __init__)
    
    def encrypt_updates(self, delta_w):  # homomorphic encrypting
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
    '''
