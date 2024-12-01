from blake3 import blake3
# nothing is needed to be changed here 
print('NOW, you are in blockchain_structure.py')


class Block:
    def __init__(self, index, previous_hash, timestamp, data, creator, nonce, hash_value):
        self.index = index # position of the block in the chain
        self.previous_hash = previous_hash # hash of the previous block
        self.timestamp = timestamp
        self.data = data # data of the block
        self.creator = creator # node that created the block
        self.nonce = nonce # nonce used in PoW
        self.hash = hash_value # hash of the current block
        print('NOW, you are in Block in blockchain_structure.py')


class Blockchain:
    def __init__(self):
        self.chain = [] # list to store the chain of blocks
        print('NOW, you are in Blockchain in blockchain_structure.py')


    def add_block(self, block):
        print('NOW, you are in add_block in blockchain_structure.py')
        if self.validate_block(block):
            self.chain.append(block) # add the block to the chain
        else:
            print(f"Invalid block index {block.index}")

    def validate_block(self, block):
        """Single source of block validation"""
        print(f"\n=== Validating block {block.index} ===")
        print(f"Creator: Node {block.creator}")
        print(f"Timestamp: {block.timestamp}")
        
        if len(self.chain) > 0:
            print("Checking block index...")
            if block.index != self.chain[-1].index + 1:
                print(f"ERROR: Invalid block index. Expected {self.chain[-1].index + 1}, got {block.index}")
                return False
            
            print("Checking previous hash...")
            if block.previous_hash != self.chain[-1].hash:
                print("ERROR: Invalid previous hash")
                return False
            
        print("Validating proof of work...")
        if not self.validate_proof_of_work(block):
            print("ERROR: Invalid proof of work")
            return False
        
        print("Block validation successful")
        return True

    def validate_proof_of_work(self, block): # method for validation PoW
        print('NOW, you are in validate_proof_of_work in blockchain_structure.py')
        block_header = f"{block.creator}{block.data}{block.previous_hash}"
        computed_hash = blake3(f"{block_header}{block.nonce}".encode()).hexdigest()
        difficulty = 4
        target = '0' * difficulty
        return computed_hash.startswith(target)

    def get_last_block(self): # get the last block in the chain
        print('NOW, you are in get_last_block in blockchain_structure.py')
        return self.chain[-1] if self.chain else None







