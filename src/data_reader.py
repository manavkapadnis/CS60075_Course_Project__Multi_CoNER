# Import the required libraries
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from global_var import wnut_iob, conll_iob

# Get Named entity recognition reader
def get_ner_reader(data):
    
    # 'fields' contains 4 lists 
    # The first list is the list of words present in the sentence
    # The last list is the list of ner tags of the words.
    
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        
        if is_divider:
            continue
        
        fields = [line.strip().split() for line in lines]
        fields = [list(field) for field in zip(*fields)]
        
        yield fields

# Function to assign the new tags 
def _assign_ner_tags(ner_tag, rep_):
    
    ner_tags_rep = []
    token_masks = []

    sub_token_len = len(rep_)
    token_masks.extend([True] * sub_token_len)
    
    if ner_tag[0] == 'B':
        
        in_tag = 'I' + ner_tag[1:]
        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    
    return ner_tags_rep, token_masks

# Function to extract spans (BI spans) and store in a dictionary
def extract_spans(tags):
    
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        
        if _cur_start is None:
            return _gold_spans
        
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        
        indicator = nt[0]
        
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        
        elif indicator == 'I':
            # do nothing
            pass
        
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    
    return gold_spans


def _is_divider(line: str) -> bool:
    
    empty_line = line.strip() == ''
    
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-" or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False

# Baseline CONLL reader class
class CoNLLReader(Dataset):
    
    def __init__(self, max_instances = -1, max_length = 50, target_vocab = None, 
                 pretrained_dir = '', encoder_model = 'xlm-roberta-large', dataset_type = 'train', language = 'english'):
        
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()['pad']
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        self.sentences = []
        self.dataset_type = dataset_type
        self.language = language

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        
        dataset_name = data if isinstance(data, str) else 'dataframe'

        print("Reading file {}".format(dataset_name))
        instance_idx = 0
        
        for fields in tqdm(get_ner_reader(data = data)):
            
            sentence_string = ' '.join(fields[0])
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_ = self.parse_line_for_ner(fields = fields)
            
            tokens_tensor = torch.tensor(tokens_sub_rep, dtype = torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype = torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)

            self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor))
            instance_idx += 1
            
            self.sentences.append(sentence_string)
        print("Finished reading {:d} instances from file {}".format(len(self.instances), dataset_name))
        df = pd.DataFrame({'sentence': self.sentences})
        df.to_csv('./' + self.language + self.dataset_type + '.csv')
    
    def parse_line_for_ner(self, fields):
        
        tokens_, ner_tags = fields[0], fields[-1]

        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        
        for idx, token in enumerate(tokens_):
            
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)
            ner_tags_rep.extend(tags)

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep = [True] * len(tokens_sub_rep)
        
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep

# Additional features CONLL class
class CoNLLReaderAdditional(Dataset):
    
    def __init__(self, max_instances = -1, max_length = 50, target_vocab = None, 
                 pretrained_dir = '', encoder_model = 'xlm-roberta-large', feature_type = 'encoded_noun_features'):
        
        self.feature_type = feature_type
        
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()['pad']
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]
    
    def give_list(self, sentence):
        res = ast.literal_eval(sentence)
        return res
    
    def get_encoded_features(self, features_data):
        
        df = pd.read_csv(features_data)
        features_list = df[self.feature_type].parallel_apply(lambda x: self.give_list(x))
        return features_list
        
    def read_data(self, data, features_data):
        
        dataset_name = data if isinstance(data, str) else 'dataframe'
        print("Obtaining {} from {}".format(self.feature_type, features_data))
        features_list = self.get_encoded_features(features_data)
        
        print("Reading file {}".format(dataset_name))
        instance_idx = 0
        
        for fields, features in tqdm(zip(get_ner_reader(data = data), features_list)):
            
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_ = self.parse_line_for_ner(fields = fields)
            
            tokens_tensor = torch.tensor(tokens_sub_rep, dtype = torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype = torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)
            features_tensor = torch.tensor(features)
            
            self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, features_tensor))
            instance_idx += 1
                    
        print("Finished reading {:d} instances from file {}".format(len(self.instances), dataset_name))
    
    def parse_line_for_ner(self, fields):
        
        tokens_, ner_tags = fields[0], fields[-1]

        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        
        for idx, token in enumerate(tokens_):
            
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)
            ner_tags_rep.extend(tags)

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep = [True] * len(tokens_sub_rep)
        
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep


def get_tagset(tagging_scheme):
    if 'conll' in tagging_scheme:
        return conll_iob
    return wnut_iob

def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large', dataset_type = 'train', language = 'english'):
    if file_path is None:
        return None
    reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, 
                         encoder_model=encoder_model, dataset_type = dataset_type, language = language)
    reader.read_data(file_path)

    return reader