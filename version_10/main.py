import asyncio
import multiprocessing as mp
import time
import signal
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import pandas as pd
from node import Node
from models import LlamaModel
from data_preparating import get_node_data, load_and_prepare_data
import random
import resource
import psutil
import torch.cuda
import gc


MAX_RUNTIME = 3600  # 1 hour in seconds

# Resource limits per core
MEMORY_LIMIT_PER_CORE = 8 * 1024 * 1024 * 1024  # 8GB in bytes
MIN_MEMORY_REQUIRED = 4 * 1024 * 1024 * 1024  # 4GB minimum

def set_process_limits():
    """Set resource limits for the current process"""
    print("\n=== Setting process resource limits ===")
    try:
        # Set memory limit
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_PER_CORE, MEMORY_LIMIT_PER_CORE))
        print(f"Memory limit set to {MEMORY_LIMIT_PER_CORE / (1024**3):.1f}GB")
        
        # Set CPU time limit
        cpu_time_limit = 3600  # 1 hour in seconds
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
        print(f"CPU time limit set to {cpu_time_limit} seconds")
        
    except Exception as e:
        print(f"WARNING: Could not set resource limits: {str(e)}")

def check_available_resources():
    """Check if sufficient resources are available"""
    print("\n=== Checking available resources ===")
    
    memory = psutil.virtual_memory()
    print(f"Total system memory: {memory.total / (1024**3):.1f}GB")
    print(f"Available memory: {memory.available / (1024**3):.1f}GB")
    
    if memory.available < MIN_MEMORY_REQUIRED:
        raise RuntimeError(f"Insufficient memory. Need at least {MIN_MEMORY_REQUIRED / (1024**3)}GB")
    
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU cores available: {cpu_count}")
    print(f"Current CPU usage: {cpu_percent}%")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU memory available: {gpu_memory / (1024**3):.1f}GB")
    
    return True

def run_node_process(node_id, num_nodes, global_model_state, nodes):
    """Run a single node process with resource management"""
    try:
        set_process_limits()
        
        # Monitor resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        print(f"\n=== Starting node {node_id} process ===")
        print(f"Initial memory usage: {initial_memory / (1024**2):.1f}MB")
        
        # Initialize node with optimized model loading
        node_data = get_node_data(node_id, num_nodes)
        node_model = load_optimized_model()  # Ensure this is called
        node_model.load_state_dict(global_model_state)
        stake = random.uniform(1, 1000)
        
        node = Node(
            node_id=node_id,
            stake=stake,
            data=node_data,
            model=node_model,
            is_malicious=(node_id < int(0.1 * num_nodes))
        )
        
        # Start node's event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run node operations
        start_time = time.time()
        metrics_history = []
        
        while time.time() - start_time < MAX_RUNTIME:
            # Participate in consensus
            loop.run_until_complete(node.participate_in_consensus(nodes))  # Pass nodes to the function
            
            # Train and collect metrics
            metrics = loop.run_until_complete(collect_node_metrics(node))
            metrics['timestamp'] = datetime.now()
            metrics['node_id'] = node_id
            metrics_history.append(metrics)
            
        return metrics_history
        
    except MemoryError:
        print(f"ERROR: Memory limit exceeded for node {node_id}")
        gc.collect()
        torch.cuda.empty_cache()
        return []
    except Exception as e:
        print(f"Error in node {node_id}: {str(e)}")
        return []

async def collect_node_metrics(node):
    """Collect metrics for a single node"""
    metrics = {
        'reputation': node.reputation,
        'blocks_created': len([b for b in node.blockchain.chain if b.creator == node.node_id]),
        'training_loss': await node.get_training_loss(),
        'consensus_participations': node.consensus_participations,
        'successful_validations': node.successful_validations
    }
    return metrics

def load_optimized_model():
    """Load model with memory optimizations"""
    print("\n=== Loading optimized model ===")
    try:
        # Use 8-bit quantization
        model = LlamaModel.from_pretrained(
            'facebook/llama-2-7b',
            load_in_8bit=True,
            device_map='auto',
            torch_dtype=torch.float16
        )
        print("Model loaded with 8-bit quantization")
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
        
        return model
    except Exception as e:
        print(f"ERROR loading optimized model: {str(e)}")
        raise

async def main():
    """Main function with resource management"""
    try:
        if not check_available_resources():
            raise RuntimeError("Resource check failed")
            
        num_cores = mp.cpu_count()
        available_memory = psutil.virtual_memory().available
        cores_to_use = min(
            num_cores,
            available_memory // MIN_MEMORY_REQUIRED
        )
        
        print(f"\nUsing {cores_to_use} cores out of {num_cores} available")
        
        num_nodes = 4
        nodes_per_core = num_nodes // num_cores
        
        # Initialize global model
        global_model = LlamaModel()
        global_model_state = global_model.state_dict()

        print('Global model installed \n')

        nodes = []  # Initialize nodes list
        for i in range(num_nodes):
            is_malicious = True if i < int(0.1 * num_nodes) else False  # 10% malicious nodes
            node_data = get_node_data(i, num_nodes)  # get the local data for the node
            node_model = LlamaModel()  # initialize the node's local model with the global model state
            node_model.load_state_dict(global_model_state)
            stake = random.uniform(1, 1000)  # assign random stake value for the nodes (in course of randomization)
            node = Node(node_id=i, stake=stake, data=node_data, model=node_model, is_malicious=is_malicious)  # created instance
            nodes.append(node)

        # Distribute data to nodes based on their capabilities
        data = load_and_prepare_data()  # Load and prepare data
        distribute_data_to_nodes(nodes, data)  # Distribute data to eligible nodes

        # Setup process pool
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Distribute nodes across cores
            futures = []
            for core in range(num_cores):
                start_node = core * nodes_per_core
                end_node = start_node + nodes_per_core if core < num_cores - 1 else num_nodes
                
                for node_id in range(start_node, end_node):
                    futures.append(
                        executor.submit(
                            run_node_process,
                            node_id,
                            num_nodes,
                            global_model_state,
                            nodes  # Pass nodes to the function
                        )
                    )
            
            # Collect results
            all_metrics = []
            for future in futures:
                try:
                    metrics = future.result()
                    all_metrics.extend(metrics)
                except Exception as e:
                    print(f"Error collecting metrics: {str(e)}")
            
            # Save metrics
            df = pd.DataFrame(all_metrics)
            df.to_csv(f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            # Analyze results
            analyze_results(df)

    except Exception as e:
        print(f"CRITICAL ERROR in main: {str(e)}")
        raise

def analyze_results(df):
    """Analyze experiment results"""
    print("\nExperiment Results:")
    
    # Consensus Analysis
    print("\nConsensus Metrics:")
    print(f"Total blocks created: {df['blocks_created'].sum()}")
    print(f"Average blocks per node: {df['blocks_created'].mean():.2f}")
    
    # Reputation Analysis
    print("\nReputation Metrics:")
    print(f"Average final reputation: {df.groupby('node_id')['reputation'].last().mean():.2f}")
    print(f"Max reputation achieved: {df['reputation'].max():.2f}")
    
    # Training Analysis
    print("\nTraining Metrics:")
    print(f"Average training loss: {df['training_loss'].mean():.4f}")
    print(f"Training loss improvement: {df.groupby('node_id')['training_loss'].apply(lambda x: x.iloc[0] - x.iloc[-1]).mean():.4f}")
    
    # Consensus Participation
    print("\nConsensus Participation:")
    print(f"Average participations: {df['consensus_participations'].mean():.2f}")
    print(f"Successful validations rate: {(df['successful_validations'] / df['consensus_participations']).mean():.2%}")

def monitor_node_behavior(nodes):
    """Monitor node behavior for anomalies."""
    print("\n=== Monitoring Node Behavior ===")
    for node in nodes:
        if node.successful_validations < 5:  # Example threshold
            print(f"WARNING: Node {node.node_id} has low successful validations.")
            # Implement actions, e.g., flag for review

def distribute_data_to_nodes(nodes, data):
    """Distribute data to nodes based on their resource capabilities."""
    print("\n=== Distributing Data to Nodes ===")
    for node in nodes:
        if node.can_load_model():  # Check if the node can load the model
            # Assign a portion of data to the node
            node_data = data[:len(data) // len(nodes)]  # Example: split data evenly
            node.data = node_data
            print(f"Data assigned to Node {node.node_id}.")
            data = data[len(node_data):]  # Remove assigned data from the pool
        else:
            print(f"Node {node.node_id} is not eligible for data assignment.")

if __name__ == '__main__':
    # Handle graceful shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Run experiment
    asyncio.run(main())
    print("Experiment completed successfully!")





