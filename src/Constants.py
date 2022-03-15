import os
import sys

import pandas as pd

THIS_DIRECTORY = os.path.dirname(__file__)
print(THIS_DIRECTORY)

PLOT_DIR = os.path.join(THIS_DIRECTORY,'../plots')

DATA_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../data/'))
DATA_PRIVACY_SECURITY_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../data/privacy_security'))
DATA_OFFICE_SUITE_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../data/office_suite'))
DATA_WEB_BROWSER_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../data/web_browser'))

RESULT_PRIVACY_SECURITY_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../results/privacy_security'))
RESULT_OFFICE_SUITE_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../results/office_suite'))
RESULT_WEB_BROWSER_DIR = os.path.normpath(os.path.join(THIS_DIRECTORY, '../results/web_browser'))

# Total App Reviews
APP_REVIEWS = os.path.join(DATA_DIR, 'df_app_reviews.csv.gzip')

"""
APP IDs
""" 
SIGNAL_ID = 'org.thoughtcrime.securesms'
WIRE_ID = 'com.wire'

FIREFOX_ID = 'org.mozilla.firefox'
BRAVE_ID = 'com.brave.browser'

NEXTCLOUD_ID = 'com.nextcloud.client'
OWNCLOUD_ID = 'com.owncloud.android'

APP_IDS = [SIGNAL_ID, WIRE_ID, FIREFOX_ID, BRAVE_ID, NEXTCLOUD_ID, OWNCLOUD_ID]

"""
App Names
"""
SIGNAL = 'Signal'
WIRE = 'Wire'
FIREFOX = 'Firefox'
BRAVE = 'BRAVE'
NEXTCLOUD = 'NEXTCLOUD'
OWNCLOUD = 'OWNCLOUD'

APPS = [SIGNAL, WIRE, FIREFOX, BRAVE, NEXTCLOUD, OWNCLOUD]

"""
Issues & Review compile setting
"""
####################################################################################
# Privacy/security focused [signal, wire]

# issues github addresses
SIGNAL_ISSUES_Repo = 'signalapp/Signal-Android'
WIRE_ISSUES_Repo = 'wireapp/wire-android' 

## Signal Issues & Reviews
SIGNAL_REVIEWS = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'df_signal_app_reviews.csv.gzip')
SIGNAL_ISSUES = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'df_signal_issues.csv.gzip')
## WIRE Issues & Reviews
WIRE_REVIEWS = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'df_wire_app_reviews.csv.gzip')
WIRE_ISSUES = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'df_wire_issues.csv.gzip')

# Signal Issues pickle file
SIGNAL_ISSUES_PICKLE = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'signal_issue.pkl')
WIRE_ISSUES_PICKLE = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'wire_issue.pkl'
) 

# Embedding files of Issues
SIGNAL_EMBEDDINGS = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'signal_embeddings_nouns.pkl')
WIRE_EMBEDDINGS = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'wire_embeddings_nouns.pkl')

# Embedding files of App reviews
SIGNAL_REVIEW_EMBEDDINGS = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'signal_review_embeddings_nouns.pkl')
WIRE_REVIEW_EMBEDDINGS = os.path.join(
    DATA_PRIVACY_SECURITY_DIR, 'wire_review_embeddings_nouns.pkl')

### Results
WIRE_SIGNAL_ISSUES_MATCHING = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'wire_signal_issues_matching.csv.gzip'
)
WIRE_SIGNAL_ISSUES_MATCHING_random = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'wire_signal_issues_matching_random.csv.gzip'
)
WIRE_SIGNAL_info = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'wire_signal_info.pkl'
)

SIGNAL_WIRE_ISSUES_MATCHING = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'signal_wire_issues_matching.csv.gzip'
)
WIRE_SIGNAL_ISSUES_MATCHING_random = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'signal_wire_issues_matching_random.csv.gzip'
)
WIRE_SIGNAL_info = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'wire_signal_info.pkl'
)

####################################################################################

##Web browsers
# issues github addresses
FIREFOX_ISSUES_Repo = 'mozilla-mobile/fenix'
BRAVE_ISSUES_Repo = 'brave/brave-browser' 

# FOREFOX ISSUES & REVIEWS
FIREFOX_ISSUES = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_issues.csv.gzip')
FIREFOX_ISSUES_PICKLE = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_issue.pkl')
FIREFOX_EMBEDDINGS = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_embeddings_nouns.pkl')
FIREFOX_REVIEWS = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_app_reviews.csv.gzip')
FIREFOX_REVIEWS_PICKLE = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review.pkl') 
FIREFOX_REVIEW_EMBEDDINGS = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_embeddings_nouns.pkl')

## 1&2
FIREFOX_ISSUES_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_issues_1.csv.gzip')
FIREFOX_EMBEDDINGS_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_embeddings_nouns_1.pkl')
FIREFOX_ISSUES_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_issues_2.csv.gzip')
FIREFOX_EMBEDDINGS_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_embeddings_nouns_2.pkl')
# FIREFOX TEST
FIREFOX_ISSUES_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_issues_test.csv.gzip')
FIREFOX_ISSUES_PICKLE_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_issue_test.pkl') 
FIREFOX_EMBEDDINGS_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_embeddings_nouns_test.pkl')
FIREFOX_REVIEWS_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_app_reviews_test.csv.gzip')
FIREFOX_REVIEWS_PICKLE_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_test.pkl') 
FIREFOX_REVIEW_EMBEDDINGS_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_embeddings_nouns_test.pkl')


# BRAVE ISSUES & REVIEWS
BRAVE_ISSUES = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_issues.csv.gzip')
BRAVE_ISSUES_PICKLE = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_issue.pkl') 
BRAVE_EMBEDDINGS = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_embeddings_nouns.pkl')
BRAVE_REVIEWS = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_app_reviews.csv.gzip')
BRAVE_REVIEWS_PICKLE = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review.pkl') 
BRAVE_REVIEW_EMBEDDINGS = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_embeddings_nouns.pkl')
# 1&2
BRAVE_ISSUES_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_issues_1.csv.gzip')
BRAVE_ISSUES_PICKLE_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_issue_1.pkl') 
BRAVE_EMBEDDINGS_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_embeddings_nouns_1.pkl')
BRAVE_REVIEWS_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_app_reviews_1.csv.gzip')
BRAVE_REVIEWS_PICKLE_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_1.pkl') 
BRAVE_REVIEW_EMBEDDINGS_1 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_embeddings_nouns_1.pkl')

BRAVE_ISSUES_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_issues_2.csv.gzip')
BRAVE_ISSUES_PICKLE_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_issue_2.pkl') 
BRAVE_EMBEDDINGS_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_embeddings_nouns_2.pkl')
BRAVE_REVIEWS_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_app_reviews_2.csv.gzip')
BRAVE_REVIEWS_PICKLE_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_2.pkl') 
BRAVE_REVIEW_EMBEDDINGS_2 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_embeddings_nouns_2.pkl')

# BRAVE TEST
BRAVE_ISSUES_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_issues_test.csv.gzip')
BRAVE_ISSUES_PICKLE_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_issue_test.pkl') 
BRAVE_EMBEDDINGS_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_embeddings_nouns_test.pkl')
BRAVE_REVIEWS_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_app_reviews_test.csv.gzip')
BRAVE_REVIEWS_PICKLE_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_test.pkl') 
BRAVE_REVIEW_EMBEDDINGS_TEST = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_embeddings_nouns_test.pkl')

### Results
BRAVE_FIREFOX_ISSUES_MATCHING = os.path.join(
    RESULT_WEB_BROWSER_DIR, 'brave_firefox_issues_matching.csv.gzip'
)
FIREFOX_BRAVE_ISSUES_MATCHING_random = os.path.join(
    RESULT_WEB_BROWSER_DIR, 'brave_firefox_issues_matching_random.csv.gzip'
)
BRAVE_FIREFOX_info = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'brave_firefox_info.pkl'
)

### Results
NEXTCLOUD_OWNCLOUD_ISSUES_MATCHING = os.path.join(
    RESULT_OFFICE_SUITE_DIR, 'nextcloud_owncloud_issues_matching.csv.gzip'
)
NEXTCLOUD_OWNCLOUD_ISSUES_MATCHING_random = os.path.join(
    RESULT_OFFICE_SUITE_DIR, 'nextcloud_owncloud_issues_matching_random.csv.gzip'
)
NEXTCLOUD_OWNCLOUD_info = os.path.join(
    RESULT_OFFICE_SUITE_DIR, 'nextcloud_owncloud_info.pkl'
)

####################################################################################

## Office suite
# issues github addresses
NEXTCLOUD_ISSUES_Repo = 'nextcloud/android'
OWNCLOUD_ISSUES_Repo = 'owncloud/android' 

## NEXTCLOUD Issues & Reviews
NEXTCLOUD_REVIEWS = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_nextcloud_app_reviews.csv.gzip')
NEXTCLOUD_ISSUES = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_nextcloud_issues.csv.gzip')
## OWNCLOUD Issues & Reviews
OWNCLOUD_REVIEWS = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_owncloud_app_reviews.csv.gzip')
OWNCLOUD_ISSUES = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_owncloud_issues.csv.gzip')

# NEXTCLOUD & OWNCLOUD Issues pickle file
NEXTCLOUD_ISSUES_PICKLE = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'nextcloud_issue.pkl')
OWNCLOUD_ISSUES_PICKLE = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'owncloud_issue.pkl'
) 

# Embedding files of Issues
NEXTCLOUD_EMBEDDINGS = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'nextcloud_embeddings_nouns.pkl')
OWNCLOUD_EMBEDDINGS = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'owncloud_embeddings_nouns.pkl')

# Embedding files of App reviews
NEXTCLOUD_REVIEW_EMBEDDINGS = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'nextcloud_review_embeddings_nouns.pkl')
OWNCLOUD_REVIEW_EMBEDDINGS = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'owncloud_review_embeddings_nouns.pkl')

### Results
OWNCLOUD_NEXTCLOUD_ISSUES_MATCHING = os.path.join(
    RESULT_WEB_BROWSER_DIR, 'owncloud_nextcloud_issues_matching.csv.gzip'
)
DATA_WOFFICE_SUITE_ISSUES_MATCHING_random = os.path.join(
    RESULT_WEB_BROWSER_DIR, 'owncloud_nextcloud_issues_matching_random.csv.gzip'
)
BRAVE_FIREFOX_info = os.path.join(
    RESULT_PRIVACY_SECURITY_DIR, 'owncloud_nextcloud_info.pkl'
)
FIREFOX_BRAVE_MATCHING_result = os.path.join(
    RESULT_WEB_BROWSER_DIR, 'firefox2brave_newbugs.csv.gzip'
)
BRAVE_FIREFOX_MATCHING_result = os.path.join(
    RESULT_WEB_BROWSER_DIR, 'brave2firefox_newbugs.csv.gzip'
)



  




# Annotation file
CODINGS = os.path.join(DATA_DIR, 'codings.csv')



class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

### Find new bugs

# FOREFOX 1-31~2-7ISSUES & REVIEWS
FIREFOX_ISSUES_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_issues01310207.csv.gzip')
FIREFOX_ISSUES_PICKLE_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_issue_01310207.pkl')
FIREFOX_EMBEDDINGS_01310207= os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_embeddings_nouns_01310207.pkl')
FIREFOX_REVIEWS_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_app_reviews_01310207.csv.gzip')
FIREFOX_REVIEWS_PICKLE_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_01310207.pkl') 
FIREFOX_REVIEW_EMBEDDINGS_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_embeddings_nouns_01310207.pkl')


# BRAVE 1-31~2-7ISSUES & REVIEWS
BRAVE_ISSUES_01310207= os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_issues_01310207.csv.gzip')
BRAVE_ISSUES_PICKLE_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_issue_01310207.pkl') 
BRAVE_EMBEDDINGS_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_embeddings_nouns_01310207.pkl')
BRAVE_REVIEWS_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_app_reviews_01310207.csv.gzip')
BRAVE_REVIEWS_PICKLE_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_01310207.pkl') 
BRAVE_REVIEW_EMBEDDINGS_01310207 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_embeddings_nouns_01310207.pkl')


# FOREFOX 2-8~3-8ISSUES & REVIEWS
FIREFOX_ISSUES_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_issues02080308.csv.gzip')
FIREFOX_ISSUES_PICKLE_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_issue_02080308.pkl')
FIREFOX_EMBEDDINGS_02080308= os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_embeddings_nouns_02080308.pkl')
FIREFOX_REVIEWS_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_firefox_app_reviews_02080308.csv.gzip')
FIREFOX_REVIEWS_PICKLE_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_02080308.pkl') 
FIREFOX_REVIEW_EMBEDDINGS_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'firefox_review_embeddings_nouns_02080308.pkl')


# BRAVE 2-8~3-8ISSUES & REVIEWS
BRAVE_ISSUES_02080308= os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_issues_02080308.csv.gzip')
BRAVE_ISSUES_PICKLE_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_issue_02080308.pkl') 
BRAVE_EMBEDDINGS_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_embeddings_nouns_02080308.pkl')
BRAVE_REVIEWS_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'df_brave_app_reviews_02080308.csv.gzip')
BRAVE_REVIEWS_PICKLE_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_02080308.pkl') 
BRAVE_REVIEW_EMBEDDINGS_02080308 = os.path.join(
    DATA_WEB_BROWSER_DIR, 'brave_review_embeddings_nouns_02080308.pkl')


# SIGNAL 2021-06-01~2022-03-10 ISSUES & REVIEWS
SIGNAL_ISSUES_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_signal_issues06010310.csv.gzip')
SIGNAL_ISSUES_PICKLE_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'signal_issue_06010310.pkl')
SIGNAL_EMBEDDINGS_06010310= os.path.join(
    DATA_OFFICE_SUITE_DIR, 'signal_embeddings_nouns_06010310.pkl')
SIGNAL_REVIEWS_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_signal_app_reviews_06010310.csv.gzip')
SIGNAL_REVIEWS_PICKLE_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'signal_review_06010310.pkl') 
SIGNAL_REVIEW_EMBEDDINGS_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'signal_review_embeddings_nouns_06010310.pkl')


# WIRE 2021-06-01~2022-03-10 ISSUES & REVIEWS
WIRE_ISSUES_06010310= os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_wire_issues_06010310.csv.gzip')
WIRE_ISSUES_PICKLE_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'wire_issue_06010310.pkl') 
WIRE_EMBEDDINGS_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'wire_embeddings_nouns_06010310.pkl')
WIRE_REVIEWS_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'df_wire_app_reviews_06010310.csv.gzip')
WIRE_REVIEWS_PICKLE_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'wire_review_06010310.pkl') 
WIRE_REVIEW_EMBEDDINGS_06010310 = os.path.join(
    DATA_OFFICE_SUITE_DIR, 'wire_review_embeddings_nouns_06010310.pkl')