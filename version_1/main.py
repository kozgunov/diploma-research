import asyncio
import random
from node import Node
from aggregator import Aggregator
from models import LLaMa3Model
from data_preparation import get_node_data, test_data_loader
from evaluation import evaluate_global_model

async def main():
    num_nodes = 101  # get started with 101 nodes
    nodes = []

    global_model = LLaMa3Model() # base model
    global_model_state = global_model.state_dict()

    for i in range(num_nodes):
        is_malicious = True if i < int(0.1 * num_nodes) else False  # 10% malicious nodes
        node_data = get_node_data(i, num_nodes)  # in order to retrieve node's local dataset
        node_model = LLaMa3Model()
        node_model.load_state_dict(global_model_state)
        stake = random.uniform(1, 10)  # make the reputation management later
        node = Node(node_id=i, stake=stake, data=node_data, model=node_model, is_malicious=is_malicious)
        nodes.append(node)

    aggregator = Aggregator(nodes)  # take aggregator from satisfying file
    epochs = 5  # may be reset and dropped out (delta-sigma)

    for epoch in range(epochs):
        print(f"--- Training epoch {epoch + 1}  started ---")
        consensus_tasks = [node.participate_in_consensus(nodes) for node in nodes]  # nodes take part in consensus for block creation
        await asyncio.gather(*consensus_tasks)

        training_tasks = [node.send_updates(aggregator) for node in nodes]  # local training and sending their updates
        await asyncio.gather(*training_tasks)

        print("Evaluating global model...")
        accuracy, perplexity = evaluate_global_model(aggregator.model, test_data_loader())
        print(f"Round {epoch + 1} Current performance: Accuracy: {accuracy:.2f}%, Perplexity: {perplexity:.2f}")

    print(f"--- training of {epochs} epoch completed ---")

if __name__ == '__main__':
    asyncio.run(main())
