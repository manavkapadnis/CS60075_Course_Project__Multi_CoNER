# Import the required libraries
import torch
import numpy as np
from torch import nn
from metric import SpanF1
from typing import OrderedDict
import torch.nn.functional as F
from transformers import AutoModel
from data_reader import extract_spans
from transformers import AutoTokenizer
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

# Baseline Model
class NERModel(nn.Module):
    
    def __init__(self,
                 lr = 1e-5,
                 dropout_rate = 0.1,
                 batch_size = 16,
                 tag_to_id = None,
                 stage = 'fit',
                 pad_token_id = 1,
                 encoder_model = 'xlm-roberta-large',
                 num_gpus = 1):
        super(NERModel, self).__init__()

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)

        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict = True)

        self.feedforward = nn.Linear(in_features = self.encoder.config.hidden_size, 
                                     out_features = self.target_size)
        
        self.crf_layer = ConditionalRandomField(num_tags = self.target_size, 
                                                constraints = allowed_transitions(constraint_type = "BIO", 
                                                                                  labels = self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()        

    def forward(self, batch):
        
        tokens, tags, token_mask, metadata = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=token_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, metadata=metadata, batch_size=batch_size)
        return output
    
    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, batch_size):
        
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)

        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        return output


# Additional Features Model
class NERModelAdditional(nn.Module):
    
    def __init__(self,
                 lr = 1e-5,
                 dropout_rate = 0.1,
                 batch_size = 16,
                 tag_to_id = None,
                 stage = 'fit',
                 pad_token_id = 1,
                 feature_length = 44,
                 encoder_model = 'xlm-roberta-large',
                 num_gpus = 1):
        super(NERModel, self).__init__()

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        self.feature_length = feature_length
        
        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict = True)
        
        self.feedforward = nn.Linear(in_features = self.encoder.config.hidden_size + self.feature_length, 
                                     out_features = self.target_size)
        
        self.crf_layer = ConditionalRandomField(num_tags = self.target_size, 
                                                constraints = allowed_transitions(constraint_type = "BIO", 
                                                                                  labels = self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()        

    def forward(self, batch):
        
        tokens, tags, token_mask, metadata, features = batch
        batch_size = tokens.size(0)
        
        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=token_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        features = features.unsqueeze_(-1)
        features = features.expand(features.shape[0], features.shape[1], embedded_text_input.shape[1])
        features = features.reshape((features.shape[0], features.shape[2], features.shape[1]))
        embedded_text_input = torch.cat((embedded_text_input, features), dim = 2)
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)
   
        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, metadata=metadata, batch_size=batch_size)
        return output
    
    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, batch_size):
        
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)

        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        return output

# Transformer-bilstm-crf model
class NERModelBiLSTMCRF(nn.Module):
    
    def __init__(self,
                 lr = 1e-5,
                 dropout_rate = 0.1,
                 batch_size = 16,
                 hidden_size = 256,
                 tag_to_id = None,
                 stage = 'fit',
                 pad_token_id = 1,
                 encoder_model = 'xlm-roberta-large',
                 num_gpus = 1):
        super(NERModel, self).__init__()

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        self.hidden_size = hidden_size

        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict = True)
        
        self.bilstm = nn.LSTM(self.encoder.config.hidden_size, self.hidden_size, 
                              batch_first = True, bidirectional = True)
        
        self.feedforward = nn.Linear(in_features = 2 * self.hidden_size, 
                                     out_features = self.target_size)
        
        self.crf_layer = ConditionalRandomField(num_tags = self.target_size, 
                                                constraints = allowed_transitions(constraint_type = "BIO", 
                                                                                  labels = self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()        

    def forward(self, batch):
        
        tokens, tags, token_mask, metadata = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=token_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.bilstm(embedded_text_input)[0]
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, metadata=metadata, batch_size=batch_size)
        return output
    
    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, batch_size):
        
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)

        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        return output

# Variable feedforward layers model
class NERModelVarFeedforward(nn.Module):
    
    def __init__(self,
                 lr = 1e-5,
                 dropout_rate = 0.1,
                 batch_size = 16,
                 tag_to_id = None,
                 stage = 'fit',
                 hidden_layer_sizes = [512, 256, 128],
                 pad_token_id = 1,
                 device = 'cuda',
                 encoder_model = 'xlm-roberta-large',
                 num_gpus = 1):
        super(NERModel, self).__init__()

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size
        self.device = device

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_list = OrderedDict()
            
        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict = True)

        self.hidden_layer_list['linear0'] = nn.Linear(in_features = self.encoder.config.hidden_size, 
                                     out_features = self.hidden_layer_sizes[0])
        
        for i in range(len(hidden_layer_sizes) - 1):
            self.hidden_layer_list['linear' + str(i+1)] = nn.Linear(in_features = self.hidden_layer_sizes[i],
                                         out_features = self.hidden_layer_sizes[i+1], device = self.device)
        
        self.hidden_layer_list['linear' + str(len(hidden_layer_sizes))] = nn.Linear(in_features = self.hidden_layer_sizes[-1],
                                        out_features = self.target_size)
        
        self.sequential_layers = nn.Sequential(self.hidden_layer_list)
        
        self.crf_layer = ConditionalRandomField(num_tags = self.target_size, 
                                                constraints = allowed_transitions(constraint_type = "BIO", 
                                                                                  labels = self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()        

    def forward(self, batch):
        
        tokens, tags, token_mask, metadata = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=token_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        token_scores = self.sequential_layers(embedded_text_input)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, metadata=metadata, batch_size=batch_size)
        return output
    
    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, batch_size):
        
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)

        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        return output