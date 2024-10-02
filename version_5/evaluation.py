import torch
import torch.nn as nn
from rouge import Rouge
import data_preparation
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu


def evaluate_global_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    predictions = []
    references = []
    all_predictions_text  = [tokens_to_words(seq) for seq in all_predictions]
    all_references_text  = [[tokens_to_words(seq[0])] for seq in all_references]
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

            for i in range(inputs.size(0)): # gathering predictions & references
                pred_tokens = data_preparation.tokenizer.decode(predicted[i], skip_special_tokens=True)
                ref_tokens = data_preparation.tokenizer.decode(labels[i], skip_special_tokens=True)
                predictions.append(pred_tokens)
                references.append(ref_tokens)

    # metrics computations

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    accuracy = (total_correct / total_tokens) * 100

    bleu_score = corpus_bleu([[ref] for ref in all_references_text], all_predictions_text) * 100

    flat_predictions = [token for seq in all_predictions_text for token in seq]
    flat_references = [token for seq in all_references_text for token in seq[0]]
    f1 = f1_score(flat_references, flat_predictions, average='weighted') * 100

    rouge = Rouge()
    rouge_scores = rouge.get_scores(all_predictions_text, all_references_text, avg=True)
    #predictions_text = [' '.join(map(str, seq)) for seq in all_predictions]
    #references_text = [' '.join(map(str, seq[0])) for seq in all_references]
    rouge_scores = rouge.get_scores(all_predictions_text, all_references_text, avg=True)

    return \
    {
        'Accuracy': accuracy,
        'Perplexity': perplexity.item(),
        'BLEU Score': bleu_score,
        'F1 Score': f1,
        'ROUGE': rouge_scores
    }


def tokens_to_words(tokens):
    #return data_preparation.tokenizer.convert_ids_to_tokens(tokens)
    return data_preparation.tokenizer.decode(tokens, skip_special_tokens=True)
