from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import optuna


class LlamaModel(nn.Module):
    def __init__(self):
        super(LlamaModel, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained('facebook/llama-2-7b') # transfer learning method implementation
        self.tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        #super(LlamaModel, self).__init__()
        #config = LlamaConfig(
        #    hidden_size=512,
        #    num_hidden_layers=6,
        #    num_attention_heads=8,
        #    intermediate_size=2048,
        #    vocab_size=32000
        #)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits


def evaluate_global_model(model):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, labels in test_data_loader():
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, -1)
            total_correct += (predicted == labels).sum().item()
            total_tokens += labels.numel()
    perplexity = np.exp(total_loss / total_tokens)
    accuracy = total_correct / total_tokens * 100
    print(f'Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.2f}%')


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.inputs.append(torch.tensor(tokens[:-1]))
            self.labels.append(torch.tensor(tokens[1:]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def get_node_data(node_id):
    # Partition the dataset
    node_texts = partitioned_texts[node_id]
    dataset = WikiTextDataset(node_texts, tokenizer)
    return dataset

def test_data_loader():
    # Create data loader for the test dataset
    dataset = WikiTextDataset(test_texts, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader



def objective(trial, train_dataset, val_dataset):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 1, 5)

    model = LlamaModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, scheduler)
        val_loss = evaluate_model(model, val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    #study = optuna.create_study(direction='minimize')
    #study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


    return val_loss
