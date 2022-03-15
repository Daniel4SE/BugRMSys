import spacy
from transformers import DistilBertModel, DistilBertTokenizer

import os 
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_spacy_model():
    return spacy.load("en_core_web_sm")

def get_distil_bert_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def get_distil_bert_model():
    return DistilBertModel.from_pretrained('distilbert-base-uncased')