import asyncio
import random
from node import Node
from aggregator import Aggregator
from models import LlamaModel
from data_preparation import get_node_data, test_data_loader, load_and_prepare_data
from evaluation import evaluate_global_model
import attack_simulation
import blockchain_structure
import consensus_mechanism
import local_training
from nltk.translate.bleu_score import corpus_bleu
from attack_simulation import Attacks_Simulation




### randomizations might be integrated in the following placements:
# 1. choice of the nodes (based on reputation, but still random)
# 2. swapping the data before its sending for local training
# 3. search for parameters, which may be started randomly






print('main.py started \n')

async def main():
    num_nodes = 101  # get started with 101 nodes
    nodes = []

    global_model = LlamaModel() # base model
    global_model_state = global_model.state_dict()
    print('model installed \n')

    for i in range(num_nodes):
        is_malicious = True if i < int(0.1 * num_nodes) else False  # 10% malicious nodes
        node_data = get_node_data(i, num_nodes)  # in order to retrieve node's local dataset
        node_model = LlamaModel()
        node_model.load_state_dict(global_model_state)
        stake = random.uniform(1, 10)  # make the reputation management later
        node = Node(node_id=i, stake=stake, data=node_data, model=node_model, is_malicious=is_malicious)
        nodes.append(node)
        if is_malicious:
            print(f" {i}-th node is appended. it's malicious node\n")
            node.update_reputations(is_malicious)
        elif not is_malicious:
            print(f" {i}-th node is appended. it's healthy node\n")
            node.update_reputations(is_malicious)

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
        metrics = evaluate_global_model(aggregator.model, test_data_loader())
        print(f"Round {epoch + 1} Metrics:")
        print(f"Accuracy: {metrics['Accuracy']:.3f}%")
        print(f"Perplexity: {metrics['Perplexity']:.3f}")
        print(f"BLEU Score: {metrics['BLEU Score']:.3f}")
        print(f"F1 Score: {metrics['F1 Score']:.3f}")
        print(f"ROUGE Scores: {metrics['ROUGE']:.3f}")

        print('--- attack simulation got started ---')
        Attacks_Simulation(nodes)
        print('--- attack simulation finished ---')

        if convergence_reached:
            break

    print(f"--- training of {epochs} epoch completed ---")

if __name__ == '__main__':
    asyncio.run(main())

print("algorithm is finished without any serious errors!")