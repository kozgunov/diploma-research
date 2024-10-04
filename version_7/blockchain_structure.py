import time
from blake3 import blake3

print('NOW, you are in aggregator.py')


class Block:
    def __init__(self, index, previous_hash, timestamp, data, creator, nonce, hash_value):
        self.index = index # position of the block in the chain
        self.previous_hash = previous_hash # hash of the previous block
        self.timestamp = timestamp 
        self.data = data # data of the block
        self.creator = creator # node that created the block
        self.nonce = nonce # nonce used in PoW
        self.hash = hash_value # hash of the current block
        print('NOW, you are in Block in aggregator.py')


class Blockchain:
    def __init__(self):
        self.chain = [] # list to store the chain of blocks
        print('NOW, you are in Blockchain in aggregator.py')


    def add_block(self, block):
        print('NOW, you are in add_block in aggregator.py')
        if self.validate_block(block):
            self.chain.append(block) # add the block to the chain
        else:
            print(f"Invalid block index {block.index}")

    def validate_block(self, block):  # validate the block 
        print('NOW, you are in validate_block in aggregator.py')
        if len(self.chain) > 0 and block.index != self.chain[-1].index + 1: # validate the index 
            return False
        if len(self.chain) > 0 and block.previous_hash != self.chain[-1].hash: # validate previous hash
            return False
        if not self.validate_proof_of_work(block): # validate the PoW
            return False
        return True
    
    def validate_proof_of_work(self, block): # method for validation PoW
        print('NOW, you are in validate_proof_of_work in aggregator.py')
        block_header = f"{block.creator}{block.data}{block.previous_hash}"
        computed_hash = blake3(f"{block_header}{block.nonce}".encode()).hexdigest()
        difficulty = 4
        target = '0' * difficulty
        return computed_hash.startswith(target)

    def get_last_block(self): # get the last block in the chain
        print('NOW, you are in get_last_block in aggregator.py')
        return self.chain[-1] if self.chain else None






    


