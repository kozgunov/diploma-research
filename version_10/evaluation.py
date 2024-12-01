import torch
import torch.nn as nn
from data_preparating import tokenizer
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

print('NOW, you are in evaluation.py')

def evaluate_global_model(model, test_loader):
    print('\n=== Starting Global Model Evaluation ===')
    model.eval()
    
    metrics = {
        'total_loss': 0.0,
        'total_correct': 0,
        'total_tokens': 0,
        'batch_count': 0
    }
    
    all_predictions = []
    all_references = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    print("Processing test batches...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            print(f"\nProcessing batch {batch_idx + 1}")
            try:
                if isinstance(inputs, list):
                    print("Padding sequences in batch...")
                    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
                    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                outputs = model(inputs)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                metrics['total_loss'] += loss.item()
                
                _, predicted = torch.max(outputs, -1)
                mask = labels != -100
                
                correct = (predicted == labels).masked_select(mask).sum().item()
                tokens = mask.sum().item()
                
                metrics['total_correct'] += correct
                metrics['total_tokens'] += tokens
                metrics['batch_count'] += 1
                
                # Collect predictions and references for BLEU and ROUGE
                all_predictions.extend(predicted[mask].cpu().numpy())
                all_references.extend(labels[mask].cpu().numpy())
                
                print(f"Batch {batch_idx + 1} - Loss: {loss.item():.4f}, Accuracy: {(correct/tokens)*100:.2f}%")
                
            except Exception as e:
                print(f"ERROR processing batch {batch_idx + 1}: {str(e)}")
                continue

    print('\nComputing final metrics...')
    try:
        results = {
            'Accuracy': (metrics['total_correct'] / metrics['total_tokens']) * 100,
            'Perplexity': torch.exp(torch.tensor(metrics['total_loss'] / metrics['total_tokens'])).item(),
            'BLEU Score': compute_bleu(all_references, all_predictions),
            'ROUGE': compute_rouge(all_references, all_predictions)
        }
        
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            
        print("=== Evaluation completed ===\n")
        return results
        
    except Exception as e:
        print(f"CRITICAL ERROR in metric computation: {str(e)}")
        return None

def compute_bleu(references, predictions):
    """Compute BLEU score."""
    # Convert predictions and references to the format required by nltk
    references = [[tokenizer.decode(ref, skip_special_tokens=True).split()] for ref in references]
    predictions = [tokenizer.decode(pred, skip_special_tokens=True).split() for pred in predictions]
    return corpus_bleu(references, predictions)

def compute_rouge(references, predictions):
    """Compute ROUGE score."""
    rouge = Rouge()
    references = [tokenizer.decode(ref, skip_special_tokens=True) for ref in references]
    predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores['rouge-l']['f']  # Return F1 score for ROUGE-L

def tokens_to_words(tokens):
    print('NOW, you are in tokens_to_words in evaluation.py')
    return tokenizer.decode(tokens, skip_special_tokens=True)
