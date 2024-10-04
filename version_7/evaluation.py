import torch
import torch.nn as nn
from rouge import Rouge
from data_preparating import tokenizer
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu

print('NOW, you are in evaluation.py')

def evaluate_global_model(model, test_loader):
    print('NOW, you are in evaluate_global_model in evaluation.py')
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    all_predictions = []
    all_references = []
    all_predictions = [tokens_to_words(seq) for seq in all_predictions]
    all_references = [[tokens_to_words(seq[0])] for seq in all_references]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
            # labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            inputs = inputs.to(model.device) # forward pass
            labels = labels.to(model.device)
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item() * inputs.size(0)
            # Get predictions
            _, predicted = torch.max(outputs, -1)
            mask = labels != -100 # create mask to ignore padding tokens
            total_correct += (predicted == labels).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            for i in range(inputs.size(0)): # gathering predictions & references
                pred_tokens = predicted[i][mask[i]].tolist()
                ref_tokens = labels[i][mask[i]].tolist()
                all_predictions.append(tokens_to_words(pred_tokens))
                all_references.append([tokens_to_words(ref_tokens)])

    print('NOW, you are computing metrics in evaluation.py')

    # metrics computation
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    accuracy = (total_correct / total_tokens) * 100
    # flat_predictions = [token for seq in all_predictions for token in seq]
    # flat_references = [token for seq in all_references for token in seq[0]]
    bleu_score = corpus_bleu(all_references, all_predictions) * 100
    flat_predictions = [word for sent in all_predictions for word in sent.split()]
    flat_references = [word for sent_list in all_references for sent in sent_list for word in sent.split()]
    f1 = f1_score(flat_references, flat_predictions, average='weighted') * 100

    rouge = Rouge()
    rouge_scores = rouge.get_scores(all_predictions, all_references, avg=True)
    # rouge_scores = rouge.get_scores(all_predictions, [ref[0] for ref in all_references], avg=True) - another computation of rouge score
    # all_predictions = [' '.join(map(str, seq)) for seq in all_predictions]
    # all_references = [' '.join(map(str, seq[0])) for seq in all_references]
    rouge_scores = rouge.get_scores(all_predictions, all_references, avg=True)

    # return metrics as dictionary for outputs
    return \
    {
        'Accuracy': accuracy,
        'Perplexity': perplexity.item(),
        'BLEU Score': bleu_score,
        'F1 Score': f1,
        'ROUGE': rouge_scores
    }


def tokens_to_words(tokens): # convert token IDs to words
    print('NOW, you are in tokens_to_words in evaluation.py')
    return tokenizer.decode(tokens, skip_special_tokens=True)


