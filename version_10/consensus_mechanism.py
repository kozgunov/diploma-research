import random
from blake3 import blake3
import time

def compute_vdf(input_value, difficulty=100000):
    """Compute Verifiable Delay Function"""
    result = input_value
    modulus = 1000000007 
    for _ in range(difficulty):
        result = pow(result, 2, modulus)
    return result

def proof_of_work(block_header, difficulty):
    """Generic PoW function"""
    nonce = 0
    target = '0' * difficulty
    while True:
        hash_result = blake3(f"{block_header}{nonce}".encode()).hexdigest()
        if hash_result.startswith(target):
            return nonce, hash_result
        nonce += 1

def calculate_selection_probability(node, all_nodes):
    """Calculate node selection probability for consensus"""
    total_stake_reputation = sum(n.stake * n.reputation for n in all_nodes)
    if total_stake_reputation == 0:
        return 1 / len(all_nodes)
    return (node.stake * node.reputation) / total_stake_reputation

def calculate_reputation_weight(node, all_nodes):
    """Calculate node's reputation-based weight for consensus"""
    total_reputation = sum(n.reputation for n in all_nodes)
    if total_reputation == 0:
        return 1 / len(all_nodes)
    return node.reputation / total_reputation

def verify_reputation(node, min_reputation=10):
    """Verify if node has sufficient reputation to participate"""
    return node.reputation >= min_reputation

def verify_vdf(input_value, output_value, difficulty, expected_time):
    """Verify VDF computation"""
    start_time = time.time()
    computed_output = compute_vdf(input_value, difficulty)
    computation_time = time.time() - start_time
    
    return (computed_output == output_value and 
            abs(computation_time - expected_time) < expected_time * 0.1)

def integrate_consensus_mechanisms(node, all_nodes):
    """Integrate all three consensus mechanisms"""
    print(f"\n=== Starting consensus integration for Node {node.node_id} ===")
    
    # PoR check
    print(f"Checking PoR for Node {node.node_id}")
    print(f"Current reputation: {node.reputation}")
    if not verify_reputation(node):
        print(f"WARNING: Node {node.node_id} failed reputation check")
        return False
    print(f"Node {node.node_id} passed reputation check")
        
    # PoT with VDF
    print(f"Starting VDF computation for Node {node.node_id}")
    try:
        vdf_input = random.randint(0, int(1e6))
        start_time = time.time()
        vdf_output = compute_vdf(vdf_input)
        vdf_time = time.time() - start_time
        print(f"VDF computation completed in {vdf_time:.2f} seconds")
    except Exception as e:
        print(f"ERROR in VDF computation: {str(e)}")
        return False
    
    if not verify_vdf(vdf_input, vdf_output, node.difficulty, expected_time=1.0):
        print(f"WARNING: VDF verification failed for Node {node.node_id}")
        return False
    print("VDF verification successful")
        
    # PoW
    print(f"Starting PoW for Node {node.node_id}")
    try:
        block_header = f"{node.node_id}{vdf_output}{node.previous_hash}"
        nonce, block_hash = proof_of_work(block_header, difficulty=4)
        print(f"PoW completed with nonce: {nonce}")
    except Exception as e:
        print(f"ERROR in PoW computation: {str(e)}")
        return False
        
    # Calculate final probability
    print("Calculating final selection probability")
    try:
        rep_weight = calculate_reputation_weight(node, all_nodes)
        time_weight = 1.0 / (1.0 + vdf_time)
        selection_probability = (rep_weight + time_weight) / 2
        print(f"Selection probability: {selection_probability:.4f}")
    except Exception as e:
        print(f"ERROR in probability calculation: {str(e)}")
        return False
    
    result = random.random() < selection_probability
    print(f"Node {node.node_id} {'selected' if result else 'not selected'} for consensus")
    return result






