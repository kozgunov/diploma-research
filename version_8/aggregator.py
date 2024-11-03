import torch
import copy
import asyncio

print('NOW, you are in aggregator.py')
# I removed all parts that involved HE
class Aggregator:
    def __init__(self, nodes):
        print('NOW, you are in __init__ in aggregator.py')
        self.nodes = nodes
        self.updates = {}  
        self.public_keys = [node.public_key for node in nodes]
        self.global_model_state = None  
        self.model = copy.deepcopy(nodes[0].model.model) 
        self.aggregation_event = asyncio.Event() 

    async def receive_updates(self, node_id, updates):
        print('NOW, you are in receive_updates in aggregator.py')
        self.updates[node_id] = updates
        if len(self.updates) == len(self.nodes): 
            await self.aggregate_updates()

    async def aggregate_updates(self):
        print('NOW, you are in aggregate_updates in aggregator.py')
        aggregated_updates = {}
        num_nodes = len(self.nodes)
        param_names = self.updates[next(iter(self.updates))].keys()

        for name in param_names:
            aggregated_updates[name] = torch.zeros_like(self.model.state_dict()[name])

        for node_id in self.updates:
            node_updates = self.updates[node_id]
            for name in param_names:
                aggregated_updates[name] += node_updates[name]

        for name in aggregated_updates:
            aggregated_updates[name] /= num_nodes

        self.update_global_model(aggregated_updates)

        for node in self.nodes:
            node.set_global_model(self.global_model_state)

        self.updates.clear()  
        self.aggregation_event.set() 

    def update_global_model(self, aggregated_updates):
        
        print('NOW, you are in update_global_model in aggregator.py')
        global_state = self.model.state_dict()  
        for name in aggregated_updates:
            global_state[name] += aggregated_updates[name]
        self.model.load_state_dict(global_state) 
        self.global_model_state = global_state 

    async def wait_for_aggregation(self):
        print('NOW, you are in wait_for_aggregation in aggregator.py')
        await self.aggregation_event.wait()
        self.aggregation_event.clear()
