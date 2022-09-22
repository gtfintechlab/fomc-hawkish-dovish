import os
import sys
from time import time
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizerFast, RobertaTokenizerFast, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

sys.path.append('..')

def train_lm_hawkish_dovish(gpu_numbers: str, train_test_data_path: str, language_model_to_use: str, seed: int, batch_size: int, learning_rate: float, save_model_path: str):
    """
    Description: Run experiment over particular batch size, learning rate and seed
    """
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_numbers)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device assigned: ", device)

    # load training data
    data_df = pd.read_excel(train_test_data_path)
    sentences = data_df['sentence'].to_list()
    labels = data_df['label'].to_numpy()

    # load tokenizer
    if language_model_to_use == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
    elif language_model_to_use == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True, do_basic_tokenize=True)
    elif language_model_to_use == 'finbert':
        tokenizer = BertTokenizerFast(vocab_file='../finbert-uncased/FinVocab-Uncased.txt', do_lower_case=True, do_basic_tokenize=True)
    elif language_model_to_use == 'flangbert':
        tokenizer = BertTokenizerFast.from_pretrained('../BERT-FLANG', do_lower_case=True, do_basic_tokenize=True)
    elif language_model_to_use == 'bert-large':
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased', do_lower_case=True, do_basic_tokenize=True)
    elif language_model_to_use == 'roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', do_lower_case=True, do_basic_tokenize=True)
    elif language_model_to_use == 'pretrain_roberta':
        tokenizer = AutoTokenizer.from_pretrained("../pretrained_roberta_output", do_lower_case=True, do_basic_tokenize=True)
    else:
        return -1

    max_length = 0
    sentence_input = []
    labels_output = []
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, str):
            tokens = tokenizer(sentence)['input_ids']
            sentence_input.append(sentence)
            max_length = max(max_length, len(tokens))
            labels_output.append(labels[i])
        else:
            pass
    
    tokens = tokenizer(sentence_input, return_tensors='pt', padding='max_length', max_length=max_length)
    labels = np.array(labels_output)

    input_ids = tokens['input_ids']
    attention_masks = tokens['attention_mask']
    labels = torch.LongTensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    test_length = int(len(dataset)  * 0.2)
    val_length = int(len(dataset) * 0.1)
    train_length = len(dataset) - val_length - test_length 
    print(f'Train Size: {train_length}, Validation Size: {val_length}, Test Size: {test_length}')
    experiment_results = []
    
    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed) 

    # select language model
    if language_model_to_use == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)
    elif language_model_to_use == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)
    elif language_model_to_use == 'finbert':
        model = BertForSequenceClassification.from_pretrained('../finbert-uncased/model', num_labels=3).to(device)
    elif language_model_to_use == 'flangbert':
        model = BertForSequenceClassification.from_pretrained('../BERT-FLANG', num_labels=3).to(device)
    elif language_model_to_use == 'bert-large':
        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=3).to(device)
    elif language_model_to_use == 'roberta-large':
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3).to(device)
    elif language_model_to_use == 'pretrain_roberta':
        model = AutoModelForSequenceClassification.from_pretrained("../pretrained_roberta_output", num_labels=3).to(device)
    else:
        return -1

    # create train-val-test split
    train, val, test = torch.utils.data.random_split(dataset=dataset, lengths=[train_length, val_length, test_length])
    dataloaders_dict = {'train': DataLoader(train, batch_size=batch_size, shuffle=True), 'val': DataLoader(val, batch_size=batch_size, shuffle=True),
    'test': DataLoader(test, batch_size=batch_size, shuffle=True)}
    print(train_length, val_length, test_length)
    # select optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    max_num_epochs = 100
    max_early_stopping = 7
    early_stopping_count = 0
    best_ce = float('inf')
    best_accuracy = float('-inf')
    best_f1 = float('-inf')

    eps = 1e-2

    for epoch in range(max_num_epochs):
        if (early_stopping_count >= max_early_stopping):
            break
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                early_stopping_count += 1
            else:
                model.eval()
            
            curr_ce = 0
            curr_accuracy = 0
            actual = torch.tensor([]).long().to(device)
            pred = torch.tensor([]).long().to(device)

            for input_ids, attention_masks, labels in dataloaders_dict[phase]:
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
                    loss = outputs.loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        curr_ce += loss.item() * input_ids.size(0)
                        curr_accuracy += torch.sum(torch.max(outputs.logits, 1)[1] == labels).item()
                        actual = torch.cat([actual, labels], dim=0)
                        pred= torch.cat([pred, torch.max(outputs.logits, 1)[1]], dim=0)
            if phase == 'val':
                curr_ce = curr_ce / len(val)
                curr_accuracy = curr_accuracy / len(val)
                currF1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
                if curr_ce <= best_ce - eps:
                    best_ce = curr_ce
                    early_stopping_count = 0
                if curr_accuracy >= best_accuracy + eps:
                    best_accuracy = curr_accuracy
                    early_stopping_count = 0
                if currF1 >= best_f1 + eps:
                    best_f1 = currF1
                    early_stopping_count = 0
                print("Val CE: ", curr_ce)
                print("Val Accuracy: ", curr_accuracy)
                print("Val F1: ", currF1)
                print("Early Stopping Count: ", early_stopping_count)
    test_ce = 0
    test_accuracy = 0
    actual = torch.tensor([]).long().to(device)
    pred = torch.tensor([]).long().to(device)
    for input_ids, attention_masks, labels in dataloaders_dict['test']:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)   
        optimizer.zero_grad()   
        with torch.no_grad():
            outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
            loss = outputs.loss
            test_ce += loss.item() * input_ids.size(0)
            test_accuracy += torch.sum(torch.max(outputs.logits, 1)[1] == labels).item()
            actual = torch.cat([actual, labels], dim=0)
            pred = torch.cat([pred, torch.max(outputs.logits, 1)[1]], dim=0)
    test_ce = test_ce / len(test)
    test_accuracy = test_accuracy/ len(test)
    test_f1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    experiment_results = [seed, learning_rate, batch_size, best_ce, best_accuracy, best_f1, test_ce, test_accuracy, test_f1]

    # save model
    if save_model_path != None:
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)

    return experiment_results


def train_lm_price_change_experiments(gpu_numbers: str, train_test_data_path: str, language_model_to_use: str):
    """
    Description: Run experiments over different batch sizes, learning rates and seeds to find best hyperparameters
    """
    results = []
    seeds = [5768, 78516, 944601]
    batch_sizes = [32, 16, 8, 4]
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]
    count = 0
    for i, seed in enumerate(seeds):
        for k, batch_size in enumerate(batch_sizes):
            for j, learning_rate in enumerate(learning_rates):

                count += 1
                print(f'Experiment {count} of {len(seeds) * len(batch_sizes) * len(learning_rates)}:')

                results.append(train_lm_hawkish_dovish(gpu_numbers, train_test_data_path, language_model_to_use, seed, batch_size, learning_rate, None))
                df = pd.DataFrame(results, columns=["Seed", "Learning Rate", "Batch Size", "Val Cross Entropy", "Val Accuracy", "Val F1 Score", "Test Cross Entropy", "Test Accuracy", "Test F1 Score"])
                df.to_excel(f'../grid_search_results/pc_split_hawkish_dovish_{language_model_to_use}.xlsx', index=False)


if __name__=='__main__':
    save_model_path = "../model_data/final_model"
    train_test_data_path = "../training_data/manual-pc-split.xlsx"
    start_t = time()
    
    # experiments
    '''
    language_model_to_use = "bert"
    train_lm_price_change_experiments(gpu_numbers="0", train_test_data_path=train_test_data_path, language_model_to_use=language_model_to_use)
    language_model_to_use = "roberta"
    train_lm_price_change_experiments(gpu_numbers="0", train_test_data_path=train_test_data_path, language_model_to_use=language_model_to_use)
    language_model_to_use = "bert-large"
    train_lm_price_change_experiments(gpu_numbers="0", train_test_data_path=train_test_data_path, language_model_to_use=language_model_to_use)
    '''
    language_model_to_use = "roberta-large"
    train_lm_price_change_experiments(gpu_numbers="0", train_test_data_path=train_test_data_path, language_model_to_use=language_model_to_use)
    '''
    language_model_to_use = "finbert"
    train_lm_price_change_experiments(gpu_numbers="1", train_test_data_path="../training_data/manual_v2.xlsx", language_model_to_use=language_model_to_use)
    language_model_to_use = "flangbert"
    train_lm_price_change_experiments(gpu_numbers="1", train_test_data_path="../training_data/manual_v2.xlsx", language_model_to_use=language_model_to_use)
    '''
    '''
    language_model_to_use = "roberta"
    output = train_lm_hawkish_dovish(gpu_numbers="0", train_test_data_path="../training_data/manual_v2.xlsx", 
    language_model_to_use=language_model_to_use, seed=5768, batch_size=8, learning_rate=1e-5, save_model_path=save_model_path)
    print(output)
    '''
    '''
    language_model_to_use = "pretrain_roberta"
    output = train_lm_hawkish_dovish(gpu_numbers="0", train_test_data_path="../training_data/manual_v2.xlsx", 
    language_model_to_use=language_model_to_use, seed=5768, batch_size=8, learning_rate=1e-5, save_model_path=None)
    print(output)
    '''
    print((time() - start_t)/60.0)