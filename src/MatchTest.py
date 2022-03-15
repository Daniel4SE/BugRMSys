from zmq import device
from torch.nn.functional import cosine_similarity
import ModelLoader
import Constants as Const
import DataLoader
from EmbeddingCalculation import textToBertEmbedding, filter_sw, filter_punc
from nltk.corpus import stopwords

import pandas as pd
from pandas.core.frame import DataFrame
import time

import os 
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOKENIZER = ModelLoader.get_distil_bert_tokenizer()
BERT_MODEL = ModelLoader.get_distil_bert_model()
SPACY_MDOEL = ModelLoader.get_spacy_model()


def calculate_distance(v1, v2):
    return cosine_similarity(v1, v2, dim=0)

def get_nearest_issues(text, embeddings, n=10):
    text = filter_punc(text)
    text = filter_sw(text, stopwords.words('english'))
    text_embedding = textToBertEmbedding(
        text, TOKENIZER, BERT_MODEL, SPACY_MDOEL)
    
    distances = []

    for issue_id, issue_embedding in embeddings.items():
        distance = calculate_distance(text_embedding, issue_embedding)
        distances.append((issue_id, distance))

    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    return distances[:n]

def print_top_N_texts(issue_input, A_issue_embeddings, B_review_embeddings, B_issue_embeddings, A_issues, B_reviews, B_issues, top_N):

    print('#'*50)
    print(f"input issue :{issue_input}")
    print(' ')
    print(" firefox relevant issues:")
    print("*"*50)


    A_issue_distances = get_nearest_issues(issue_input, A_issue_embeddings, top_N)

    for issue_id, A_issue_distance in A_issue_distances:
        A_issue_title = A_issues.loc[A_issues['id'] == issue_id]['title'].values[0]
        A_issue_created_time = A_issues.loc[A_issues['id'] == issue_id]['created_at'].values[0]
        A_issue_url = A_issues.loc[A_issues['id'] == issue_id]['url'].values[0]
        print(f"{A_issue_distance:>.2f}: {A_issue_title}: {A_issue_created_time}: {A_issue_url}")


    print('#'*50)
    print(f"input: firefoxissue :{issue_input}")
    print(' ')
    print(" relevant reviews in brave:")
    print('#'*50)

    B_review_distances = get_nearest_issues(issue_input, B_review_embeddings, top_N)

    for reviewId, B_review_distance in B_review_distances:
        B_review_title = B_reviews.loc[B_reviews['reviewId'] == reviewId]['content'].values[0]
        B_review_created_time = B_reviews.loc[B_reviews['reviewId'] == reviewId]['at'].values[0]
        print(f"{B_review_distance:>.2f}: {B_review_title}: {B_review_created_time}")
    

    print('#'*50)
    print(f"input: firefox issue :{issue_input}")
    print(' ')
    print(" relevant issues in brave:")
    print('#'*50)

    B_issue_distances = get_nearest_issues(issue_input, B_issue_embeddings, top_N)
    for issue_id, B_issue_distance in B_issue_distances:
        B_issue_title = B_issues.loc[B_issues['id'] == issue_id]['title'].values[0]
        B_issue_created_time = B_issues.loc[B_issues['id'] == issue_id]['created_at'].values[0]
        B_issue_url = B_issues.loc[B_issues['id'] == issue_id]['url'].values[0]
        print(f"{B_issue_distance:>.2f}: {B_issue_title}: {B_issue_created_time}: {B_issue_url}")
    


if __name__ == "__main__":
    top_N = 20 # relevant size
    A_issue_embeddings = DataLoader.load_embeddings(
        Const.FIREFOX_EMBEDDINGS_02080308)
    B_review_embeddings = DataLoader.load_embeddings(
        Const.BRAVE_REVIEW_EMBEDDINGS_02080308)
    B_issue_embeddings = DataLoader.load_embeddings(
        Const.BRAVE_EMBEDDINGS_02080308)    
    # load issues and reviews
    A_issues = DataLoader.load_df_compressed(Const.FIREFOX_ISSUES_02080308)
    B_reviews = DataLoader.load_df_compressed(Const.BRAVE_REVIEWS_02080308)
    B_issues = DataLoader.load_df_compressed(Const.BRAVE_ISSUES_02080308)
    

    issue_input = ' default browser every day.' 
    issue_input_time = 'sss' #"2022/1/16 10:51:21"
    print_top_N_texts(issue_input, A_issue_embeddings, B_review_embeddings, B_issue_embeddings, A_issues, B_reviews, B_issues, top_N)



































    """
    wire_reviews_datafram = DataFrame(wire_reviews_list_returns, columns = ['distance', 'content', 'at'])
    wire_reviews_datafram = wire_reviews_datafram.sort_values(by="at")
    for index, row in wire_reviews_datafram.iterrows():
        print(f"{row['distance']:>.2f}, {row['content']}, {row['at']}")
    """

