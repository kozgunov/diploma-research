from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
import optuna

print('NOW, you are  in models.py')


class LlamaModel(nn.Module):
    def __init__(self):
        super(LlamaModel, self).__init__() # load the pre-trained LLaMA-2 model for transfer learning
        self.model = LlamaForCausalLM.from_pretrained('facebook/llama-2-7b') # transfer learning method implementation
        self.tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-2-7b') # corresponding tokenizator to llama2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device to GPU (obv.)
        self.model.to(self.device) # same
        super(LlamaModel, self).__init__()

        #config = LlamaConfig(hidden_size=512,num_hidden_layers=6,num_attention_heads=8,intermediate_size=2048,vocab_size=32000)

    def forward(self, input_ids, attention_mask=None):
        print('NOW, you are in forward in models.py')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) # perform a forward pass through the model
        logits = outputs.logits # got non-normalized predictions
        return logits


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True) # tokenize the text (we may remove additional atribute)
            if len(tokens) > 1: # prepare inputs and labels by shifting tokens
                self.inputs.append(torch.tensor(tokens[:-1]))
                self.labels.append(torch.tensor(tokens[1:]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def get_node_data(node_id, num_nodes, train_texts, tokenizer): # partition the dataset among nodes
    print('NOW, you are in get_node_data in models.py')
    total_texts = len(train_texts)
    partition_size = total_texts // num_nodes
    start_idx = node_id * partition_size
    end_idx = start_idx + partition_size if node_id < num_nodes - 1 else total_texts
    node_texts = train_texts[start_idx:end_idx]
    dataset = WikiTextDataset(node_texts, tokenizer) # create a dataset for the node
    return dataset


def test_data_loader(test_texts, tokenizer): # function to create a test data loader
    print('NOW, you are in test_data_loader in models.py')
    dataset = WikiTextDataset(test_texts, tokenizer) # compose a dataset for test data
    loader = DataLoader(dataset, batch_size=32, shuffle=False) # data for batching
    return loader


def evaluate_model(model, val_loader, loss_fn): # evaluate w.r.t. loss(and avg_loss)
    print('NOW, you are in evaluate_model in models.py')
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


def objective(trial, train_dataset, val_dataset): # define the objective function for Optuna hyperparameter optimization
    print('NOW, you are in objective in models.py')
    val_loss = None
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 1, 5)
    model = LlamaModel() # initialize the base model here
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # set the optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) # set the schedular
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create data loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # set the loss function
    for epoch in range(num_epochs): # training
        train_model(model, train_loader, optimizer, scheduler, loss_fn) # train&evaluate the model
        val_loss = evaluate_model(model, val_loader, loss_fn)
        trial.report(val_loss, epoch) # Optuna usage
        if trial.should_prune(): # pruning
            raise optuna.exceptions.TrialPruned()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    best_trial = study.best_trial # check it carefully (helpful tool for the future usage as grid)
    print("Best trial:", best_trial)
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    return val_loss


def train_model(model, train_loader, optimizer, scheduler, loss_fn): # functions for training and evaluation
    print('NOW, you are in train_model in models.py')
    model.train() # training. this function is also used in main.py (we can remove one of them for coherence)
    for inputs, labels in train_loader:
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()






