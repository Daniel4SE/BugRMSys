import MatchTest
# import Constants as Const
import ConstantsRQ4 as Const
import DataLoader
from tqdm import tqdm
import pandas as pd
from pandas.core.frame import DataFrame
import pickle
import numpy
import time
import numpy as np
np.random.seed(2022) # seed: 2020, 2021, 2022

def RQ1_results_all(A_ISSUES, B_ISSUES, A_ISSUES_EMBEDDING, B_ISSUES_EMBEDDING, B_A_ISSUES_MATCHING):

    # parameters
    score_boardline = 0.86
    issue_length = 5
    #process_bar = ShowProcess(max_steps, 'preprocess finished, Okay!')
    

    # RQ1: How many sharing issues in same-domain apps (wire & signal)
    # load signal issues & telegram reviews
    A_issues = DataLoader.load_df_compressed(A_ISSUES)
    B_issues = DataLoader.load_df_compressed(B_ISSUES)

    A_issue_embeddings = DataLoader.load_embeddings(
        A_ISSUES_EMBEDDING)
    B_issue_embeddings = DataLoader.load_embeddings(
        B_ISSUES_EMBEDDING)

    wi_sg_result_data = {}

    A_issue_length = A_issues.shape[0]
    B_issue_length = B_issues.shape[0]

    wi_sg_result_data['A_issue_length'] = A_issue_length
    wi_sg_result_data['B_issue_length'] =  B_issue_length

    DataLoader.save_embeddings(Const.WIRE_SIGNAL_info, wi_sg_result_data)


    process_bar = Const.ShowProcess(B_issues.shape[0], 'preprocess finished, Okay!')
    wi_sg_result_list = []
    for i, row in B_issues.iterrows():
        
        B_issue_id = row['id']
        B_issue_text = row['title']
        B_issue_body = row['body']
        B_create_time = row['created_at']
        #ts_sg_result_list_element['telegram', issue_id, issue_text, create_time]
        A_issue_distances = MatchTest.get_nearest_issues(B_issue_text, A_issue_embeddings, 3)
        
        wi_sg_result_list_columns = ['wire_issue_id', 'wire_issue_text', 'wire_issue_body', 'wire_create_time', \
            'distance', 'relevant_issue', 'signal_issue_id', 'signal_issue_text', 'signal_issue_body', 'signal_create_time']
        for issue_id, distance in A_issue_distances[:3]:
            if distance > score_boardline:
                wi_sg_result_list_element = []

                A_issue_id = A_issues.loc[A_issues['id'] == issue_id]['id'].values[0]
                A_issue_text = A_issues.loc[A_issues['id'] == issue_id]['title'].values[0]
                A_issue_body = A_issues.loc[A_issues['id'] == issue_id]['body'].values[0]
                A_create_time = A_issues.loc[A_issues['id'] == issue_id]['created_at'].values[0]

                wi_sg_result_list_element.append(B_issue_id)
                wi_sg_result_list_element.append(B_issue_text)
                wi_sg_result_list_element.append(B_issue_body)                
                wi_sg_result_list_element.append(B_create_time)
                wi_sg_result_list_element.append(numpy.around(distance.detach().numpy(), 4))
                wi_sg_result_list_element.append('relevant_issue')
                wi_sg_result_list_element.append(A_issue_id)
                wi_sg_result_list_element.append(A_issue_text)
                wi_sg_result_list_element.append(A_issue_body)
                wi_sg_result_list_element.append(A_create_time)

                wi_sg_result_list.append(wi_sg_result_list_element)
                #print('ts_sg_result_list_element:', ts_sg_result_list_element)
        process_bar.show_process()
        time.sleep(0.01)        

    wire_signal_issues = DataFrame(wi_sg_result_list, columns= wi_sg_result_list_columns) 
    DataLoader.save_df_compressed(B_A_ISSUES_MATCHING, wire_signal_issues) 

def random_results_for_manual(random_num):
    df = DataLoader.load_df_compressed(Const.WIRE_SIGNAL_ISSUES_MATCHING)
    dataset_length = df.shape[0]
    indices = np.random.permutation(dataset_length)
    wi_sg_result_random150 = df.iloc[indices[:random_num]]
    DataLoader.save_df_compressed(Const.WIRE_SIGNAL_ISSUES_MATCHING_random, wi_sg_result_random150)
    print(indices[:3])

def RQ1_results_all_random(A, B, A_EMBEDDING, B_EMBEDDING, B_A_MATCHING):

    # parameters
    score_boardline = 0.89
    #process_bar = ShowProcess(max_steps, 'preprocess finished, Okay!')
    

    # RQ1: How many sharing issues in same-domain apps (wire & signal)
    # load signal issues & telegram reviews
    A_issues = DataLoader.load_df_compressed(A)
    B_issues = DataLoader.load_df_compressed(B)

    A_issue_embeddings = DataLoader.load_embeddings(
        A_EMBEDDING)
    B_issue_embeddings = DataLoader.load_embeddings(
        B_EMBEDDING)

   

    indexes = np.arange(A_issues.shape[0])
    #np.random.shuffle(indexes)
    #A_issues = A_issues.loc[indexes[:300]]




    print(A_issues.shape[0])
    print("#"*100)
    process_bar = Const.ShowProcess(A_issues.shape[0], 'preprocess finished, Okay!')


    wi_sg_result_list = []
    for i, row in A_issues.iterrows():
        
        A_issue_id = row['id']
        A_issue_text = row['title']
        A_issue_body = row['body']
        A_create_time = row['created_at']
        #ts_sg_result_list_element['telegram', issue_id, issue_text, create_time]
        B_issue_distances = MatchTest.get_nearest_issues(A_issue_text, B_issue_embeddings, 1)

        print(B_issue_distances)
        
        wi_sg_result_list_columns = ['firefox_issue_id', 'firefox_issue_text', 'firefox_issue_body', 'firefox_create_time', \
            'distance', 'brave_review_id', 'brave_review_text', 'brave_create_time']
        
        for issue_id, distance in B_issue_distances[:1]:
            if distance > score_boardline:
                wi_sg_result_list_element = []
                B_issue_id = B_issues.loc[B_issues['reviewId'] == issue_id]['reviewId'].values[0]
                B_issue_text = B_issues.loc[B_issues['reviewId'] == issue_id]['content'].values[0]
                B_create_time = B_issues.loc[B_issues['reviewId'] == issue_id]['at'].values[0]

                wi_sg_result_list_element.append(A_issue_id)
                wi_sg_result_list_element.append(A_issue_text)
                wi_sg_result_list_element.append(A_issue_body)                
                wi_sg_result_list_element.append(A_create_time)
                wi_sg_result_list_element.append(numpy.around(distance.detach().numpy(), 4))
                wi_sg_result_list_element.append(B_issue_id)
                wi_sg_result_list_element.append(B_issue_text)
                wi_sg_result_list_element.append(B_create_time)

                wi_sg_result_list.append(wi_sg_result_list_element)
            #print('ts_sg_result_list_element:', ts_sg_result_list_element)
        process_bar.show_process()

    wire_signal_issues = DataFrame(wi_sg_result_list, columns= wi_sg_result_list_columns) 
    DataLoader.save_df_compressed(B_A_MATCHING, wire_signal_issues) 

def get_issue_time_distribution(issue_gzip_file, plot_file = None):
    issues = DataLoader.load_df_compressed(issue_gzip_file)

    print(issues['created_at'].values[-1])

if __name__ == "__main__":

    # step 1, return pairs <SIGNAL_ISSUES, WIRE_ISSUES> by machine, need to chech by manual, we need to randomly choose 150 items of it by step 2
    A = Const.OWNCLOUD_ISSUES
    B = Const.NEXTCLOUD_REVIEWS
    A_EMBEDDING = Const.OWNCLOUD_EMBEDDINGS
    B_EMBEDDING = Const.NEXTCLOUD_REVIEW_EMBEDDINGS
    B_A_MATCHING = Const.NEXTCLOUD_OWNCLOUD_ISSUES_MATCHING_random
    RQ1_results_all_random(A, B, A_EMBEDDING, B_EMBEDDING, B_A_MATCHING)


    # step 2 get random_results 150 from wire_signal_issues
    #random_results_for_manual(100)
    #get_issue_time_distribution(Const.SIGNAL_ISSUES)
