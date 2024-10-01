import random
from hashlib import sha256


async def participate_in_consensus(self): # PoS
    total_stake = sum(node.stake for node in self.nodes)
    selection_probability = self.stake / total_stake
    if random.random() < selection_probability:
        await self.create_block()


async def create_block(self): # simulate VDF computation for PoT
    vdf_input = random.randint(0, 1e6)
    self.vdf_output = self.compute_vdf(vdf_input) # minimal usage of PoW
    nonce = 0
    target = 1e5  # adjust difficulty later
    while int(sha256(f"{self.vdf_output}{nonce}".encode()).hexdigest(), 16) > target:
        nonce += 1

    # broadcast of the blocks
    block = \
    {
        'creator': self.node_id,
        'vdf_output': self.vdf_output,
        'nonce': nonce, # DO NOT TOUCH IT
        'transactions': [],  # for the future application
        'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None,
    }
    block['hash'] = sha256(str(block).encode()).hexdigest()
    for node in self.nodes:
        node.receive_block(block)

def compute_vdf(self, input_value):
    result = input_value
    for _ in range(100000):  # time delay
        result = pow(result, 2, 1e9 + 7)
    return result
