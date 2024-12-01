import torch
import copy
import asyncio
import hashlib

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
        self.initial_data_hashes = {}  # Store initial data hashes for verification

    async def receive_updates(self, node_id, updates, initial_data):
        print('NOW, you are in receive_updates in aggregator.py')
        print(f"Receiving updates from Node {node_id}...")
        
        # Compute hash of the initial data
        initial_data_hash = self.compute_data_hash(initial_data)
        self.initial_data_hashes[node_id] = initial_data_hash
        
        self.updates[node_id] = updates
        
        if len(self.updates) == len(self.nodes):
            print("All updates received. Starting aggregation...")
            await self.aggregate_updates()

    def compute_data_hash(self, data):
        """Compute a hash for the given data."""
        data_str = str(data)  # Convert data to string representation
        return hashlib.sha256(data_str.encode()).hexdigest()  # Compute SHA-256 hash

    def verify_data_integrity(self, node_id, current_data):
        """Verify that the data for a node has not changed after training."""
        print(f"Verifying data integrity for Node {node_id}...")
        current_data_hash = self.compute_data_hash(current_data)
        
        if current_data_hash != self.initial_data_hashes[node_id]:
            print(f"WARNING: Data for Node {node_id} has changed after training!")
            return False
        print(f"Data for Node {node_id} is unchanged.")
        return True

    async def aggregate_updates(self):
        print('\n=== Starting update aggregation ===')
        print(f"Number of updates received: {len(self.updates)}/{len(self.nodes)}")
        
        if not self.updates:
            print("WARNING: No updates received")
            return
        
        try:
            aggregated_updates = {}
            num_nodes = len(self.nodes)
            param_names = self.updates[next(iter(self.updates))].keys()
            print(f"Aggregating {len(param_names)} parameter groups")

            for name in param_names:
                print(f"\nProcessing parameter group: {name}")
                aggregated_updates[name] = torch.zeros_like(self.model.state_dict()[name])
                
                for node_id in self.updates:
                    try:
                        node_updates = self.updates[node_id]
                        # Verify data integrity before aggregation
                        if not self.verify_data_integrity(node_id, self.nodes[node_id].data):
                            print(f"Skipping aggregation for Node {node_id} due to data integrity issues.")
                            continue
                        
                        aggregated_updates[name] += node_updates[name]
                        print(f"Added updates from Node {node_id}")
                    except Exception as e:
                        print(f"ERROR processing updates from Node {node_id}: {str(e)}")

                aggregated_updates[name] /= num_nodes
                print(f"Averaged updates for {name}")

            print("\nUpdating global model...")
            self.update_global_model(aggregated_updates)
            print("Global model updated successfully")

            print("\nDistributing new model to nodes...")
            for node in self.nodes:
                node.set_global_model(self.global_model_state)
                print(f"Model distributed to Node {node.node_id}")

            self.updates.clear()
            print("Update buffer cleared")
            self.aggregation_event.set()
            print("=== Aggregation completed ===\n")
            
        except Exception as e:
            print(f"CRITICAL ERROR in aggregation: {str(e)}")
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
