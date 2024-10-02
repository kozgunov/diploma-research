

class Block:
    def __init__(self, index, previous_hash, timestamp, data, creator, nonce, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.creator = creator
        self.nonce = nonce
        self.hash = hash

class Blockchain:
    def __init__(self):
        self.chain = []

    def add_block(self, block):
        if self.validate_block(block):
            self.chain.append(block)
        else:
            print(f"Invalid block index {block.index}")

    def validate_block(self, block):  # validate block index
        if len(self.chain) > 0 and block.index != self.chain[-1].index + 1: # validate previous hash
            return False
        if len(self.chain) > 0 and block.previous_hash != self.chain[-1].hash: # additional validations (e.g., PoW, signatures)
            return False
        return True

    def get_last_block(self):
        return self.chain[-1] if self.chain else None

