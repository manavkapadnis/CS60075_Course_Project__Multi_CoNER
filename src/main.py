import os
import torch
import warnings
import numpy as np
from utils import Args, dataloading, train_dataloader, val_dataloader, train_and_evaluate
from models import NERModel, NERModelAdditional, NERModelBiLSTMCRF, NERModelVarFeedforward

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    # Create an object of arguments class
    sg = Args()

    # Load the train and dev data 
    train_data, dev_data = dataloading(sg)

    # Create the training and validation dataloader in PyTorch format
    training_dataloader = train_dataloader(sg, train_data)
    validation_dataloader = val_dataloader(sg, dev_data)

    # Create the model
    model = NERModel(tag_to_id = train_data.get_target_vocab(), dropout_rate = sg.dropout, 
                    batch_size = sg.batch_size, stage = sg.stage, lr = sg.lr, feature_length = sg.feature_length,
                    encoder_model = sg.encoder_model, num_gpus = sg.gpus).to(sg.device)

    # Specify the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=sg.lr)

    # Call the function train and evaluate
    train_and_evaluate(training_dataloader, validation_dataloader, model, 
                        optimizer, sg, additional_features = sg.additional_features)

if __name__ == '__main__':
    main()