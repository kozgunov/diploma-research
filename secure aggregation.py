class Aggregator:
    def __init__(self, nodes):
        self.nodes = nodes
        self.encrypted_updates = {}
        self.public_keys = [node.public_key for node in nodes]
        self.global_model_state = None  # Global model parameters
        self.model = nodes[0].model  # Reference model structure

    async def receive_updates(self, node_id, encrypted_updates):
        self.encrypted_updates[node_id] = encrypted_updates
        if len(self.encrypted_updates) == len(self.nodes):
            await self.aggregate_updates()

    async def aggregate_updates(self):
        # Aggregate encrypted updates
        aggregated_updates = {}
        for name in self.encrypted_updates[self.nodes[0].node_id]:
            encrypted_sum = None
            for node_id in self.encrypted_updates:
                encrypted_update = self.encrypted_updates[node_id][name]
                if encrypted_sum is None:
                    encrypted_sum = encrypted_update
                else:
                    encrypted_sum = [
                        x + y for x, y in zip(encrypted_sum, encrypted_update)
                    ]
            aggregated_updates[name] = encrypted_sum
        # Decrypt the aggregated updates (requires private key)
        decrypted_updates = self.decrypt_updates(aggregated_updates)
        # Update the global model
        self.update_global_model(decrypted_updates)
        # Distribute the new global model to nodes
        for node in self.nodes:
            node.set_global_model(self.global_model_state)
        # Clear stored updates for next round
        self.encrypted_updates.clear()

    def decrypt_updates(self, aggregated_updates):
        decrypted_updates = {}
        for name, encrypted_params in aggregated_updates.items():
            decrypted = [self.nodes[0].private_key.decrypt(x) / len(self.nodes)
                         for x in encrypted_params]
            param_shape = self.model.state_dict()[name].shape
            decrypted_param = torch.tensor(decrypted).view(param_shape)
            decrypted_updates[name] = decrypted_param
        return decrypted_updates

    def update_global_model(self, decrypted_updates):
        # Update the global model parameters
        global_state = self.model.state_dict()
        for name in decrypted_updates:
            global_state[name] += decrypted_updates[name]
        self.global_model_state = global_state
        self.model.load_state_dict(global_state)
