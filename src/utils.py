# Import the required libraries
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_reader import get_reader, get_tagset

# Arguments class
class Args():
    
    def __init__(self):
        
        self.train = '../input/semeval-task-11/EN-English/en_train.conll'
        self.test = '../input/semeval-task-11/EN-English/en_dev.conll'
        self.dev = '../input/semeval-task-11/EN-English/en_dev.conll'
        
        self.train_features = '../input/semeval-task-11/en_train_features.csv'
        self.test_features = '../input/semeval-task-11/en_dev_features.csv'
        self.dev_features = '../input/semeval-task-11/en_dev_features.csv'
        
        # Feature type will be either 'encoded_noun_features' or 'encoded_dependency_features'
        self.feature_type = 'encoded_dependency_features'
        self.feature_length = 44
        
        self.out_dir = './'
        self.iob_tagging = 'wnut'
        
        self.max_instances = -1
        self.max_length = 50
        # here if you are implemeting the additional features model
        # then set additional_features as True
        self.additional_features = False

        self.hidden_layer_sizes = [512, 256, 128]

        self.encoder_model = 'bert-base-multilingual-cased'
        self.model = './'
        self.model_name = 'bert-base-multilingual-cased'
        self.stage = 'fit'
        self.prefix = 'test'

        self.batch_size = 32
        self.gpus = 1
        self.device = 'cuda'
        self.epochs = 3
        self.lr = 1e-5
        self.dropout = 0.1
        self.max_grad_norm = 1.0

# Load the data
def dataloading(sg):
    train_data = get_reader(file_path=sg.train, target_vocab=get_tagset(sg.iob_tagging), 
                            encoder_model=sg.encoder_model, max_instances=sg.max_instances,
                            max_length=sg.max_length, dataset_type = 'train', language = 'hindi_')
    dev_data = get_reader(file_path=sg.dev, target_vocab=get_tagset(sg.iob_tagging), 
                          encoder_model=sg.encoder_model, max_instances=sg.max_instances, 
                          max_length=sg.max_length, dataset_type = 'dev', language = 'hindi_')

    return train_data, dev_data

# Collate batch for dataloader
def collate_batch(batch):
        
        batch_ = list(zip(*batch))
        tokens, masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size = (len(tokens), max_len), 
                                   dtype = torch.long).fill_(1)
        tag_tensor = torch.empty(size = (len(tokens), max_len), 
                                 dtype = torch.long).fill_(model.tag_to_id['O'])
        mask_tensor = torch.zeros(size = (len(tokens), max_len), dtype = torch.bool)

        for i in range(len(tokens)):
            
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]

        return token_tensor, tag_tensor, mask_tensor, gold_spans

# Get train dataloader and validation dataloader
def train_dataloader(sg, train_data):
    loader = DataLoader(train_data, batch_size = sg.batch_size, collate_fn = collate_batch, num_workers = 1)
    return loader

def val_dataloader(sg, dev_data):
    loader = DataLoader(dev_data, batch_size = sg.batch_size, collate_fn = collate_batch, num_workers = 1)
    return loader

# Function to train and evaluate the model
def train_and_evaluate(training_dataloader, validation_dataloader, model, optimizer, sg, additional_features = False):
    
    print("----------------------- Training ----------------------------")
    print()
    
    # Training loop
    for epoch_i in tqdm(range(sg.epochs)):

        epoch_iterator = tqdm(training_dataloader, desc = "Iteration", position = 0, leave = True)

        # Train the model
        model.train()
        training_loss = 0

        for step, batch in enumerate(epoch_iterator):

            if additional_features:
                batch = (batch[0].to(sg.device), batch[1].to(sg.device), batch[2].to(sg.device), batch[3], batch[4].to(sg.device))
            else:
                batch = (batch[0].to(sg.device), batch[1].to(sg.device), batch[2].to(sg.device), batch[3])
            
            # forward pass
            output = model.forward(batch)

            # backward pass
            loss = output['loss']
            loss.backward()

            # track train loss
            training_loss += loss.item()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = sg.max_grad_norm)

            # update parameters
            optimizer.step()

        # print train loss per epoch
        training_loss = training_loss / len(training_dataloader)
        print()
        print('Epoch: {} \tTraining Loss: {:.5f}'.format(epoch_i + 1, training_loss))
    
        metric_scores = model.span_f1.get_metric()
        model.span_f1.reset()
        
        print()
        print("Epoch: {} metrics".format(epoch_i+1))
        print()
        for key, value in metric_scores.items():
            print("{}: {:.5f},".format(key, value), end = " ")
        print()
    
    print()
    print("--------------------- Evaluation ---------------------")
    print()
    
    # Loop for evaluation on validation set
    
    epoch_iterator = tqdm(validation_dataloader, desc = "Iteration", position = 0, leave = True)
    
    validation_loss = 0
    for step, batch in enumerate(epoch_iterator):
        
        if additional_features:
            batch = (batch[0].to(sg.device), batch[1].to(sg.device), batch[2].to(sg.device), batch[3], batch[4].to(sg.device))
        else:
            batch = (batch[0].to(sg.device), batch[1].to(sg.device), batch[2].to(sg.device), batch[3])
        
        with torch.no_grad():
            output = model.forward(batch)

        loss = output['loss']
        validation_loss += loss.item()

    validation_loss = validation_loss / len(validation_dataloader)
    print()
    print('Validation Loss: {:.5f}'.format(validation_loss))
    print()
    metric_scores = model.span_f1.get_metric()
    model.span_f1.reset()
    print()
    print("Metrics on validation set")
    print()
    for key, value in metric_scores.items():
        print("{}: {:.5f},".format(key, value), end = " ")
    print()
    print()
    torch.save(model, "./" + sg.model_name + "_" + str(sg.batch_size) + "_" + str(sg.lr) + ".pt")
    print("Saved the model")
