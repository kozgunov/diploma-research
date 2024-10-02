import asyncio
import random
from node import Node
from aggregator import Aggregator
from models import LlamaModel
from data_preparation import get_node_data, test_data_loader, load_and_prepare_glue, load_and_prepare_superglue
from evaluation import evaluate_global_model
from attack_simulation import Attacks_Simulation, model_poisoning_attack


### randomizations might be integrated in the following placements:
# 1. choice of the nodes (based on reputation, but still random)
# 2. swapping the data before its sending for local training
# 3. search for parameters, which may be started randomly - also initialization may be random
# 4. attack simulations
# probably, nothing more here


def calculate_epochs(num_nodes, max_epochs=5,
                     min_epochs=1):  # number of epochs has to depend on number of nodes for even(uniform) training
    epochs = max(min_epochs, int(round(max_epochs * (50 / num_nodes))))
    return epochs


def has_converged(previous_metrics, current_metrics, threshold=0.01):
    delta = abs(current_metrics['Perplexity'] - previous_metrics['Perplexity'])
    return delta < threshold


print('main.py started \n')

num_nodes = 101  # get started with 101 nodes --> then we will create the algorithm for adding users automatically


async def main():
    nodes = []

    global_model = LlamaModel()  # base model
    global_model_state = global_model.state_dict()
    print('model installed \n')

    for i in range(num_nodes):
        is_malicious = True if i < int(0.1 * num_nodes) else False  # 10% malicious nodes
        node_data = get_node_data(i, num_nodes)  # in order to retrieve node's local dataset
        node_model = LlamaModel()
        node_model.load_state_dict(global_model_state)
        stake = random.uniform(1, 1000)  # make the reputation management later
        node = Node(node_id=i, stake=stake, data=node_data, model=node_model, is_malicious=is_malicious)
        nodes.append(node)
        if node.is_malicious:
            model_poisoning_attack(node)
            print(f" {i}-th node is appended. it's malicious node\n")
            node.update_reputations(node, 'malicious')
        elif not node.is_malicious:
            print(f" {i}-th node is appended. it's healthy node\n")
            node.update_reputations(node, 'honest')

    aggregator = Aggregator(nodes)  # take aggregator from satisfying file
    try:
        epochs = calculate_epochs(num_nodes)
    except:
        epochs = 5  # may be reset and dropped out (delta-sigma)

    for epoch in range(epochs):
        print(f"--- Training epoch {epoch + 1}  started ---")
        consensus_tasks = [node.participate_in_consensus(nodes) for node in
                           nodes]  # nodes take part in consensus for block creation
        await asyncio.gather(*consensus_tasks)

        training_tasks = [node.send_updates(aggregator) for node in nodes]  # local training and sending their updates
        await asyncio.gather(*training_tasks)

        print("Evaluating global model...")

        previous_metrics = None
        for round_num in range(epochs):  # the convergent of the model
            accuracy, perplexity = evaluate_global_model(aggregator.model, test_data_loader())
            metrics = evaluate_global_model(aggregator.model, test_data_loader())
            print(f"Round {epoch + 1} Metrics:")
            print(f"Accuracy: {metrics['Accuracy']:.3f}%")
            print(f"Perplexity: {metrics['Perplexity']:.3f}")
            print(f"BLEU Score: {metrics['BLEU Score']:.3f}")
            print(f"F1 Score: {metrics['F1 Score']:.3f}")
            print(f"ROUGE Scores: {metrics['ROUGE']:.3f}")
            if previous_metrics and has_converged(previous_metrics, metrics):
                print(f"Convergence reached at round {round_num + 1} ---> update the global model")
                break
            previous_metrics = metrics  # then we update the global model

        print('--- attack simulation got started ---')  # try to attack all or not or the nodes
        Attacks_Simulation(nodes)
        print('--- attack simulation finished ---')

    print(f"--- training of {epochs} epoch completed ---")


if __name__ == '__main__':
    asyncio.run(main())

print("algorithm is finished without any serious errors!")
