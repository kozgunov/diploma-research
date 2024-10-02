import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import data_preparation

def evaluate_global_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    predictions = []
    references = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, -1)
            mask = labels != -100
            total_correct += (predicted == labels).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()
            for i in range(inputs.size(0)):
                pred_tokens = data_preparation.tokenizer.decode(predicted[i], skip_special_tokens=True)
                ref_tokens = data_preparation.tokenizer.decode(labels[i], skip_special_tokens=True)
                predictions.append(pred_tokens)
                references.append(ref_tokens)
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    accuracy = (total_correct / total_tokens) * 100
    bleu_score = sentence_bleu.corpus_bleu([[ref] for ref in references], predictions) # Calculate BLEU
    rouge = Rouge() # Calculate ROUGE
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    return accuracy, perplexity.item(), bleu_score, rouge_scores
