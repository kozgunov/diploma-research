import torch
import torch.nn as nn
from rouge import Rouge
from data_preparating import tokenizer
from nltk.translate.bleu_score import corpus_bleu

print('NOW, you are in evaluation.py')

def evaluate_global_model(model, test_loader):
    print('NOW, you are in evaluate_global_model in evaluation.py')
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    all_predictions = []  
    all_references = []  
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    with torch.no_grad(): 
        for inputs, labels in test_loader:
            if isinstance(inputs, list):
                # Pad sequences to the same length
                inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
                labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(inputs)

            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(outputs, -1)

            mask = labels != -100
            total_correct += (predicted == labels).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            for i in range(inputs.size(0)):
                pred_tokens = predicted[i][mask[i]].tolist()
                ref_tokens = labels[i][mask[i]].tolist()
                pred_words = tokens_to_words(pred_tokens)
                ref_words = tokens_to_words(ref_tokens)
                # For BLEU, append lists of tokens
                all_predictions.append(pred_words.split())
                all_references.append([ref_words.split()])  # List of references per sample

    print('NOW, you are computing metrics in evaluation.py')

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    accuracy = (total_correct / total_tokens) * 100

    bleu_score = corpus_bleu(all_references, all_predictions) * 100

    all_predictions_text = [' '.join(tokens) for tokens in all_predictions]
    all_references_text = [' '.join(ref_tokens[0]) for ref_tokens in all_references]  # Since we have one reference per sample

    rouge = Rouge()
    rouge_scores = rouge.get_scores(all_predictions_text, all_references_text, avg=True)
    
 # Oliseenko commented that it's better to apply metrics, which are recommended/published as the best for NAMELY THIS DATASET
 # I think it is not a good idea first because of the resean that I explaiined in "data_preprocessing.py" file, second because we have to generalize our system not to restrict it to a particular dataset
    
    return {
        'Accuracy': accuracy,
        'Perplexity': perplexity.item(),
        'BLEU Score': bleu_score,
        'ROUGE': rouge_scores
    }

def tokens_to_words(tokens):
    print('NOW, you are in tokens_to_words in evaluation.py')
    return tokenizer.decode(tokens, skip_special_tokens=True)
