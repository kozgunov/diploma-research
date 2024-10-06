import asyncio
import random
from node import Node
from aggregator import Aggregator
from models import LlamaModel
from data_preparating import get_node_data, test_data_loader, load_and_prepare_data, load_and_prepare_glue, load_and_prepare_superglue
from evaluation import evaluate_global_model
from attack_simulation import AttacksSimulation
from blockchain_structure import Blockchain, Block


print('the program started, libraries correctly installed')


# number of epochs has to depend on number of nodes for even(uniform) training
def calculate_epochs(num_nodes, max_epochs=5, min_epochs=1):
    epochs = max(min_epochs, int(round(max_epochs * (50 / num_nodes))))
    return epochs


# returning True (improve more than 0.05(threshold) / 5(metrics) * 100(%) = 1%), it will approve the convergent of model
def has_converged(previous_metrics, current_metrics, threshold=0.05):
    delta_perp = abs(current_metrics['Perplexity'] - previous_metrics['Perplexity'])
    delta_acc = abs(current_metrics['Accuracy'] - previous_metrics['Accuracy'])
    delta_f1 = abs(current_metrics['F1 Score'] - previous_metrics['F1 Score'])
    delta_bleu = abs(current_metrics['BLEU Score'] - previous_metrics['BLEU Score'])
    delta_rouge = abs(current_metrics['ROUGE'] - previous_metrics['ROUGE'])
    delta = delta_rouge + delta_bleu + delta_acc + delta_f1 + delta_perp
    if delta >= threshold:
        return True
    else:
        return False


print('main.py started \n')


async def main():
    nodes = []
    num_nodes = 101  # get started with 101 nodes --> then we will create the algorithm for adding users automatically
    global_model = LlamaModel()  # base model
    global_model_state = global_model.state_dict()
    previous_metrics = None

    print('global model installed \n')

    for i in range(num_nodes):
        is_malicious = True if i < int(0.1 * num_nodes) else False  # 10% malicious nodes
        node_data = get_node_data(i, num_nodes)  # get the local data for the node
        node_model = LlamaModel() # initialize the node's local model with the global model state
        node_model.load_state_dict(global_model_state)
        stake = random.uniform(1, 1000)  # assign random stake value for the nodes (in course of randomization)
        node = Node(node_id=i, stake=stake, data=node_data, model=node_model, is_malicious=is_malicious) # created instance
        nodes.append(node)
        if node.is_malicious: # update node reputation based on its behavior
            AttacksSimulation(node)
            print(f" {i}-th node is appended. it's malicious node\n")
            node.update_reputations(node, 'malicious')
        elif not node.is_malicious:
            print(f" {i}-th node is appended. it's healthy node\n")
            node.update_reputations(node, 'honest')

    aggregator = Aggregator(nodes)  # take aggregator from satisfying file
    print('Aggregator initialized.\n')
    #try:
    #    epochs = calculate_epochs(num_nodes) # depends on amount of nodes
    #except:
    epochs = 1  # may be stopped in advance

    # Oliseenko advised to use solely 1 epoch for the post-training, and if it's not enough - work deeper with datasets

    # also Oliseenko said that full training process better doing in main.py rather than in another files

    for epoch in range(epochs):
        print(f"--- Training epoch {epoch + 1}  started ---")
        consensus_tasks = [node.participate_in_consensus(nodes) for node in nodes]  # nodes take part in consensus for block creation
        await asyncio.gather(*consensus_tasks)
        training_tasks = [node.local_training(epochs=epochs) for node in nodes]  # local training and sending their updates (idk if it's encrypted)
        await asyncio.gather(*training_tasks)

        await aggregator.wait_for_aggregation() # wait for aggregator to aggregate updates and update the global model
        print("Evaluating global model...")



        for round_num in range(epochs):  # the convergent of the model under threshold and so on...
            metrics = evaluate_global_model(aggregator.model, test_data_loader())
            print(f"Round {epoch + 1} Metrics:")
            print(f"Accuracy: {metrics['Accuracy']:.3f}%")
            print(f"Perplexity: {metrics['Perplexity']:.3f}")
            print(f"BLEU Score: {metrics['BLEU Score']:.3f}")
            print(f"F1 Score: {metrics['F1 Score']:.3f}")
            print(f"ROUGE Scores: {metrics['ROUGE']:.3f}")
            if previous_metrics and has_converged(previous_metrics, metrics):
                print(f"Convergence reached at round {round_num + 1} ---> update the global model")
                previous_metrics = metrics # set the goal for the future model's update
                break
            previous_metrics = metrics  # update the global model or goes up to the end

        print('--- attack simulation got started ---')  # try to attack all or not or the nodes
        attack_simulator = AttacksSimulation(nodes)
        attack_simulator.simulate_attacks()
        print('--- attack simulation finished ---')

    print(f"--- training of {epochs + 1} epoch completed ---")


if __name__ == '__main__':
    asyncio.run(main())
    print("algorithm is finished without any serious errors!")







