import pandas as pd
from google_play_scraper import Sort, reviews, reviews_all

import Constants as Const
import DataLoader
import datetime

def get_reviews_all(app_id):
    us_reviews = reviews_all(
        app_id,
        sleep_milliseconds=0, # defaults to 0
        lang='en', # defaults to 'en'
        country='us', # defaults to 'us'
        sort= Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
    )
    for r in us_reviews:
        #print(r)
        r['app_id'] = app_id

    return us_reviews

def returndf(APP_reviews, column_names):
    App_reviews_list = []
    for i in range(len(APP_reviews)):
        reviewId = APP_reviews[i]['reviewId']
        userName = APP_reviews[i]['userName']
        content = str(APP_reviews[i]['content'])
        score = APP_reviews[i]['score']
        reviewCreatedVersion = APP_reviews[i]['reviewCreatedVersion']
        at = APP_reviews[i]['at']
        replyContent = APP_reviews[i]['replyContent']
        repliedAt = APP_reviews[i]['repliedAt']
        app_id = APP_reviews[i]['app_id']

        App_reviews_list_item = [reviewId, userName, content, score, reviewCreatedVersion, at, replyContent, repliedAt, app_id]
        App_reviews_list.append(App_reviews_list_item)
      
  
    df_file = pd.DataFrame(App_reviews_list, columns=Column_names)
    return df_file
def dffilter(df_1):
    indexes = []
    for i in range(len(df_1['content'])):
        if len(str(df_1['content'][i]).split(' '))<=10 or len(str(df_1['content'][i]).split(' '))>200:
            indexes.append(i)
    print(indexes)

    print(df_1.shape[0])
    df_1 = df_1.drop(indexes)
    return df_1


if __name__ == "__main__": 
    sdate = '20210601'
    edate = '20220310'
    APP_ID = Const.NEXTCLOUD_ID
    APP_REVIEWS = Const.NEXTCLOUD_REVIEWS


    APP_reviews = get_reviews_all(APP_ID)
    Column_names = ['reviewId', 'userName', 'content', 'score', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'app_id']
    df_1 = returndf(APP_reviews, Column_names)
    df_1 = dffilter(df_1)
    DataLoader.save_df_compressed(APP_REVIEWS, df_1)
    print(df_1.shape[0])
    

    # # Step 3: choose date from 2020-01-01 ~ 2022-01-24
    # df_1 = DataLoader.load_df_compressed(APP_REVIEWS)
    # print("***")
    # print(df_1.shape[0])
    # s_date  = datetime.datetime.strptime(sdate,'%Y%m%d').date()
    # #m_date  = datetime.datetime.strptime('20210101','%Y%m%d').date()
    # e_date  = datetime.datetime.strptime(edate,'%Y%m%d').date()

    # for i in range(df_1.shape[0]):
    #     df_1['at'][i] = str(df_1['at'][i]).split(' ')[0]
    #     df_1['at'][i] = datetime.datetime.strptime(df_1['at'][i], '%Y-%m-%d').date()
    # df_01 = df_1[(df_1['at'] >= s_date) & ((df_1['at'] <= e_date))]
    # #df_02 = df_1[(df_1['at'] >= m_date) & ((df_1['at'] <= e_date))]
    # print(df_01.shape[0])

    # #DataLoader.save_df_compressed(Const.BRAVE_REVIEWS, df_1)
    # #df_01 = df_1.sort_index('at')
    # DataLoader.save_df_compressed(APP_REVIEWS, df_01)
    # #DataLoader.save_df_compressed(Const.BRAVE_REVIEWS_2, df_02)


    


    








    



