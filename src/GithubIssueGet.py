from github import Github
import os
from pprint import pprint
import pickle
# import Constants as Const
import ConstantsRQ4 as Const
import DataLoader

from collections import Counter
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
import datetime

class issuesget:
    """
    parameters:     namerepo-> username + reponame, for example: "signalapp/Signal-Android"
                    state-> open, close, all, etc
    input:          issues_save(issues, file_tosave_path)
    output:         a pickle file of issues
    """
    def __init__(self, namerepo, state):
        self.token = os.getenv('GITHUB_TOKEN','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')
        self.namerepo = namerepo
        self.state = state
    
    def issues_get(self):
        g = Github(self.token)
        repo = g.get_repo(self.namerepo)
        issues = repo.get_issues(state = self.state)
        return issues
    
    def issues_save(self, issues, file):
        issuesfile = []
        for issue in issues:
            print(issue)
            issuesfile.append(issue)

        with open(file, 'wb') as f:
            pickle.dump(issuesfile, f)
        return True

def return_issue_lists(app_issues):
    list_items = []
    for i, issue in enumerate(tqdm(app_issues)):
        list_item = [issue.url, issue.repository, issue.labels_url, issue.comments_url, \
            issue.events_url, issue.html_url, issue.id, issue.number, \
                issue.title, issue.user, issue.labels, issue.state, issue.milestone, \
                    issue.comments, issue.created_at, issue.updated_at, \
                        issue.closed_at, issue.body, issue.pull_request]
        list_items.append(list_item)
    column_names = ['url', 'repository', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'number', 'title', 'user', 'labels', \
        'state', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'body', 'pull_request']
    return list_items, column_names


if __name__ == '__main__':


    APP_ISSUES_Repo = Const.NEXTCLOUD_ISSUES_Repo
    APP_ISSUE_PICKLE = Const.NEXTCLOUD_ISSUES_PICKLE
    APP_ISSUES = Const.NEXTCLOUD_ISSUES
    

    # Step 1: Get issue and save to pickle file
    namerepo = APP_ISSUES_Repo #
    getissue = issuesget(namerepo, state = "all")
    issues = getissue.issues_get()
    getissue.issues_save(issues, APP_ISSUE_PICKLE)
    


    # Step 2: load pickle file and extract title, created_at, etc information and generate .csv.gzip file
    file =  open(APP_ISSUE_PICKLE,'rb')
    app_issues = pickle.load(file)
    print("#"*50)
    list_items, column_names = return_issue_lists(app_issues)
    df = DataFrame(list_items, columns= column_names)    
    df_open = df.loc[df['state'] == 'open']
    df_closed = df.loc[df['state'] == 'closed']
    df = pd.concat([df_open, df_closed], axis=0)
    print(df.columns)
    df = df.sort_values(by= 'created_at')
    print(df.shape[0])
    print(Counter(df['state'].values))
    DataLoader.save_df_compressed(APP_ISSUES, df)

    

    # Step 3: choose date from 2020-01-01 ~ 2022-01-01
    df = DataLoader.load_df_compressed(APP_ISSUES)
    print(df.shape[0])
    s_date  = datetime.datetime.strptime('20220208','%Y%m%d').date()
    #m_date  = datetime.datetime.strptime('20210101','%Y%m%d').date()
    e_date  = datetime.datetime.strptime('20220308','%Y%m%d').date()

    for i in range(df.shape[0]):
        df['created_at'][i] = df['created_at'][i].split(' ')[0]
        df['created_at'][i] = datetime.datetime.strptime(df['created_at'][i], '%Y-%m-%d').date()
    #df_01 = df[(df['created_at'] >= s_date) & ((df['created_at'] <= m_date))]
    df_02 = df[(df['created_at'] >= s_date) & ((df['created_at'] <= e_date))]
    print(df_02.shape[0])

    #DataLoader.save_df_compressed(Const.BRAVE_ISSUES_TEST, df_01)
    DataLoader.save_df_compressed(APP_ISSUES, df_02)
    #print(df_01.shape[0], df_02.shape[0])
    
    #print(df.shape[0])
    #DataLoader.save_df_compressed(APP_ISSUES, df)