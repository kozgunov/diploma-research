async def participate_in_consensus(self):
    # PoR/PoS: node selection based on stake
    total_stake = sum(node.stake for node in self.nodes)
    selection_probability = self.stake / total_stake
    if random.random() < selection_probability: # selected for block creation
        await self.create_block() 

async def create_block(self):
    vdf_input = random.randint(0, 1e6) # simulate VDF computation for PoT
    self.vdf_output = self.compute_vdf(vdf_input)
    nonce = 0
    target = 1e5  # adjust difficulty as needed
    while int(sha256(f"{self.vdf_output}{nonce}".encode()).hexdigest(), 16) > target:
        nonce += 1
        
    # create and broadcast block
    block = {
        'creator': self.node_id,
        'vdf_output': self.vdf_output,
        'nonce': nonce,
        'transactions': [],  # add transactions if applicable
        'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None,
    }
    block['hash'] = sha256(str(block).encode()).hexdigest()
    for node in self.nodes:
        node.receive_block(block)

def compute_vdf(self, input_value):
    # simplified VDF computation
    result = input_value
    for _ in range(100000):  # adjust iterations to simulate time delay
        result = pow(result, 2, 1e9 + 7)
    return result
