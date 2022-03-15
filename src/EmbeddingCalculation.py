from importlib.resources import contents
import logging
from sys import maxsize
from unittest.util import _MAX_LENGTH

import spacy_alignments as tokenizations
from tqdm import tqdm
import pandas as pd

# import Constants as Const
import ConstantsRQ4 as Const
import DataLoader
import ModelLoader
import re
import nltk
from nltk.corpus import stopwords

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import os 
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time
from time import sleep



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def filter_sw(text, sw_nltk):
    words = [word for word in text.split() if word.lower() not in sw_nltk]
    new_text = " ".join(words)
    return new_text

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）)：]+", " ", sentence)  
    sentence = sentence.replace('=',' ')
    sentence = sentence.replace('{',' ')
    sentence = sentence.replace(']',' ')
    sentence = sentence.replace('}',' ')
    sentence = sentence.replace('[',' ')
    sentence = sentence.replace('<',' ')
    sentence = sentence.replace('>',' ')
    sentence = sentence.replace('-',' ')
    sentence = sentence.replace(':',' ')
    sentence = sentence.replace(';',' ')
    return(sentence)


def textToBertEmbedding(text: str, bert_tokenizer, bert_model, spacy_model):
    text = str(text)
    bert_tokens = bert_tokenizer.tokenize(text)


    doc = spacy_model(text)
    spacy_tokens = [token.text for token in doc]
    s2b, _ = tokenizations.get_alignments(spacy_tokens, bert_tokens)

    index_noun_tokens = []

    for ent, token_mapping in zip(doc, s2b):
        if ent.pos_ == 'NOUN':
            index_noun_tokens += token_mapping
    
   
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True)
    
    """
    
    if inputs['input_ids'].shape[1]>500:
        print(inputs['input_ids'].shape[1], inputs['attention_mask'].shape[1])
        print(inputs)
        inputs['input_ids'] = inputs['input_ids'][:,:200]
        inputs['attention_mask'] = inputs['attention_mask'][:,:200]
        print(inputs)
    """

       
    

    outputs = bert_model(**inputs)    

    last_hidden_states = outputs[0]
    vectors = last_hidden_states[0]


    vectors_without_special_tokens = vectors[1:-1]
    del vectors

    if index_noun_tokens:
        vectors = vectors_without_special_tokens[index_noun_tokens]
    else:
        vectors = vectors_without_special_tokens

    return vectors.mean(axis=0)

# word --> embedding &&& save embed.pkl
def embed(file_issue_embeddings, texts, ids, tokenizer, model, spacy_model, limit):
    number_texts = len(texts)
    embeddings = DataLoader.load_embeddings(file_issue_embeddings)

    if limit:
        texts = texts[:limit]
        ids = ids[:limit]
    new_embeddings = 0

    for ids, texts in tqdm(zip(ids, texts), total=number_texts):
        starttime = time.time()
        if ids not in embeddings:            
            issue_embedding = textToBertEmbedding(
                texts, tokenizer, model, spacy_model)
            embeddings[ids] = issue_embedding
            new_embeddings += 1
            del issue_embedding
            torch.cuda.empty_cache()

            
        endtime =time.time()
        timespend = endtime - starttime
        print("time spend:", timespend)

    if new_embeddings > 0:
        DataLoader.save_embeddings(file_issue_embeddings, embeddings)
    else:
        logger.info(f'no new embedding for {file_issue_embeddings}')


def embed_signal_issues(file_issue_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding signal issues...')
    signal = DataLoader.load_df_compressed(Const.SIGNAL_ISSUES)
    issue_descriptions = signal['title'].to_list()
    for i in range(len(issue_descriptions)):
        issue_descriptions[i] = filter_sw(issue_descriptions[i], stopwords.words('english'))
        issue_descriptions[i] = filter_punc(issue_descriptions[i])
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = signal['id'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_issue_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_wire_issues(file_issue_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding wire issues...')
    wire = DataLoader.load_df_compressed(Const.WIRE_ISSUES)
    issue_descriptions = wire['title'].to_list()
    for i in range(len(issue_descriptions)):
        issue_descriptions[i] = filter_sw(issue_descriptions[i], stopwords.words('english'))
        issue_descriptions[i] = filter_punc(issue_descriptions[i])
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = wire['id'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_issue_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_wire_reviews(file_review_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding wire review...')
    wire = DataLoader.load_df_compressed(Const.WIRE_REVIEWS)
    issue_descriptions = wire['content'].to_list()
    for i in range(len(issue_descriptions)):
        issue_descriptions[i] = filter_sw(issue_descriptions[i], stopwords.words('english'))
        issue_descriptions[i] = filter_punc(issue_descriptions[i])
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = wire['reviewId'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_review_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_firefox_issues(file_issue_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding firefox issues...')
    telegram = DataLoader.load_df_compressed(Const.FIREFOX_ISSUES_02080308)
    issue_descriptions = telegram['title'].to_list()
    for line in issue_descriptions:
        line = filter_sw(line, stopwords.words('english'))
        line = filter_punc(line)
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = telegram['id'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_issue_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_firefox_reviews(file_review_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding firefox review...')
    firefox = DataLoader.load_df_compressed(Const.FIREFOX_REVIEWS_02080308)
    issue_descriptions = firefox['content'].to_list()
    for i in range(len(issue_descriptions)):
        issue_descriptions[i] = filter_sw(issue_descriptions[i], stopwords.words('english'))
        issue_descriptions[i] = filter_punc(issue_descriptions[i])
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = firefox['reviewId'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_review_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_brave_issues(file_issue_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding brave issues...')
    telegram = DataLoader.load_df_compressed(Const.BRAVE_ISSUES_02080308)
    issue_descriptions = telegram['title'].to_list()
    for line in issue_descriptions:
        line = filter_sw(line, stopwords.words('english'))
        line = filter_punc(line)
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = telegram['id'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_issue_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_brave_reviews(file_review_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding brave review...')
    brave = DataLoader.load_df_compressed(Const.BRAVE_REVIEWS_02080308)
    issue_descriptions = brave['content'].to_list()
    for i in range(len(issue_descriptions)):
        issue_descriptions[i] = filter_sw(issue_descriptions[i], stopwords.words('english'))
        issue_descriptions[i] = filter_punc(issue_descriptions[i])
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = brave['reviewId'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_review_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_nextcloud_issues(file_issue_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding nextcloud issues...')
    telegram = DataLoader.load_df_compressed(Const.NEXTCLOUD_ISSUES)
    issue_descriptions = telegram['title'].to_list()
    for line in issue_descriptions:
        line = filter_sw(line, stopwords.words('english'))
        line = filter_punc(line)
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = telegram['id'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_issue_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_nextcloud_reviews(file_review_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding nextcloud review...')
    brave = DataLoader.load_df_compressed(Const.NEXTCLOUD_REVIEWS)
    issue_descriptions = brave['content'].to_list()
    for i in range(len(issue_descriptions)):
        issue_descriptions[i] = filter_sw(issue_descriptions[i], stopwords.words('english'))
        issue_descriptions[i] = filter_punc(issue_descriptions[i])
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = brave['reviewId'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_review_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)

def embed_owncloud_issues(file_issue_embeddings, tokenizer, model, spacy_model, limit):
    logger.info('Embedding owncloud issues...')
    telegram = DataLoader.load_df_compressed(Const.OWNCLOUD_ISSUES)
    issue_descriptions = telegram['title'].to_list()
    for line in issue_descriptions:
        line = filter_sw(line, stopwords.words('english'))
        line = filter_punc(line)
    print("#"*100)
    print(issue_descriptions[:10])
    # pre-process, remove punctuation, etc.

    issue_ids = telegram['id'].to_list()
    print(issue_ids[0])
    print(type(issue_ids), len(issue_ids))

    embed(file_issue_embeddings, issue_descriptions, issue_ids,
          tokenizer, model, spacy_model, limit)


if __name__ == "__main__":

    ### filter reviews which length < 10

    """
    telegram_reviews = DataLoader.load_df_compressed(Const.TELEGRAM_REVIEWS)
    indexs = []
    for index, row in telegram_reviews.iterrows():
        if len(str(row.content).split())<10:
            indexs.append(index)

    telegram_reviews = telegram_reviews.drop(indexs)
    telegram_reviews['content'] = str(telegram_reviews['content'])
    print(telegram_reviews.columns)

    columns = ["reviewId", "content", "score", "at"]
    datarewritelist = []
    for index, row in telegram_reviews.iterrows():
        list_add = []
        list_add.append(index)
        list_add.append(row.content)
        list_add.append(row.score)
        list_add.append(row.at)
        datarewritelist.append(list_add)
    
    df = pd.DataFrame(datarewritelist, columns= columns)

    
    DataLoader.save_df_compressed(Const.TELEGRAM_REVIEWS, df) 

    review_ids = telegram_reviews['reviewId'].to_list()
    print("data lines: ",len(review_ids))
    """




    tokenizer = ModelLoader.get_distil_bert_tokenizer()
    model = ModelLoader.get_distil_bert_model()
    spacy_model = ModelLoader.get_spacy_model()

    #embed_signal_issues(Const.SIGNAL_EMBEDDINGS, tokenizer, model, spacy_model, None)
    #embed_wire_issues(Const.WIRE_EMBEDDINGS, tokenizer, model, spacy_model, None)
    #embed_firefox_issues(Const.FIREFOX_EMBEDDINGS_02080308, tokenizer, model, spacy_model, None)
    #embed_firefox_reviews(Const.FIREFOX_REVIEW_EMBEDDINGS_02080308, tokenizer, model, spacy_model, None)
    #embed_brave_issues(Const.BRAVE_EMBEDDINGS_02080308, tokenizer, model, spacy_model, None)
    #embed_brave_reviews(Const.BRAVE_REVIEW_EMBEDDINGS_02080308, tokenizer, model, spacy_model, None)
    #embed_nextcloud_issues(Const.NEXTCLOUD_EMBEDDINGS, tokenizer, model, spacy_model, None)
    #embed_nextcloud_reviews(Const.NEXTCLOUD_REVIEW_EMBEDDINGS, tokenizer, model, spacy_model, None)
    embed_owncloud_issues(Const.OWNCLOUD_EMBEDDINGS, tokenizer, model, spacy_model, None)
    #embed_wire_reviews(Const.WIRE_REVIEW_EMBEDDINGS, tokenizer, model, spacy_model, None)
