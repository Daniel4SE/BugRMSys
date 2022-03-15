""""
Data analysis, such as: recover-ratio, wordcloud, feature distribution, and so on.
"""
from http.client import CONFLICT
from unittest import result
import DataLoader
import Constants as Const
import EmbeddingCalculation as EC

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
import matplotlib.pyplot as plt

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.corpus import stopwords

np.random.seed(2020) # seed: 2020, 2021, 2022

def statistic(signal_issues, wire_issues, most_common = None):
    #print(signal_issues.columns, len(signal_issues.columns), " ", wire_issues.columns, len(wire_issues.columns))
    signal_title_text = ' '.join(signal_issues['title'].values)
    signal_title_text = EC.filter_punc(signal_title_text)
    signal_title_text = EC.filter_sw(signal_title_text, stopwords.words('english'))
    signal_title_text = signal_title_text.replace('-', ' ')
    signal_title_text = signal_title_text.replace(':', ' ')
    signal_title_text = signal_title_text.replace('[', ' ')
    signal_title_text = str.lower(signal_title_text) 
    signal_title_text = ''.join([i for i in signal_title_text if not i.isdigit()])
    
    
    wire_title_text = ' '.join(wire_issues['title'].values)
    wire_title_text = EC.filter_punc(wire_title_text)
    wire_title_text = EC.filter_sw(wire_title_text, stopwords.words('english'))
    wire_title_text = wire_title_text.replace('-', ' ')
    wire_title_text = wire_title_text.replace(':', ' ')
    wire_title_text = wire_title_text.replace('[', ' ')
    wire_title_text = str.lower(wire_title_text)
    wire_title_text = ''.join([i for i in wire_title_text if not i.isdigit()])

    
    signal_counter = Counter(signal_title_text.split(' '))
    signal_most = signal_counter.most_common(most_common)
    signal_most_values = []
    signal_most_nums = []
    for i in range(len(signal_most)):
        signal_most_values.append(signal_most[i][0])
        signal_most_nums.append(signal_most[i][1])

    wire_counter = Counter(wire_title_text.split(' '))
    wire_most = wire_counter.most_common(most_common)
    wire_most_values = []
    wire_most_nums = []
    for i in range(len(wire_most)):
        wire_most_values.append(wire_most[i][0])
        wire_most_nums.append(wire_most[i][1])

    set_signal = set(signal_most_values)
    set_wire = set(wire_most_values)

    overlap = set_signal & set_wire
    universe = set_signal | set_wire
    
    tosignal = np.around(float(len(overlap))/ len(set_signal), 4)
    towire = np.around(float(len(overlap))/ len(set_wire), 4)
    touniverse = np.around(float(len(overlap))/len(universe), 4)

    results = [tosignal, towire, touniverse]

    return signal_counter, wire_counter, signal_title_text, wire_title_text, (results, overlap), (signal_most_values, signal_most_nums, wire_most_values, wire_most_nums)

def all_results(namelist, most_common = None):
    privacy_security = namelist[0]
    web_browser = namelist[1]
    office_suite = namelist[2]

    signal_issues = privacy_security[0]
    wire_issues = privacy_security[1]

    firefox_issues = web_browser[0]
    brave_issues = web_browser[1]

    nextcloud_issues = office_suite[0]
    owncloud_issues = office_suite[1]

    signal_counter, wire_counter, signal_title_text, wire_title_text, s_w_results, s_wplot_data = statistic(signal_issues, wire_issues, most_common)
    firefox_counter, brave_counter, firefox_title_text, signal_title_text, f_b_results, f_b_plot_data = statistic(firefox_issues, brave_issues, most_common)
    nextcloud_counter, owncloud_counter, nextcloud_title_text, owncloud_title_text, n_o_results, n_o_plot_data = statistic(nextcloud_issues, owncloud_issues, most_common)

    results = [(signal_counter, wire_counter, signal_title_text, wire_title_text, s_w_results, s_wplot_data), \
        (firefox_counter, brave_counter, firefox_title_text, signal_title_text, f_b_results, f_b_plot_data), \
            (nextcloud_counter, owncloud_counter, nextcloud_title_text, owncloud_title_text, n_o_results, n_o_plot_data)]
    
    return results


    

if __name__ == "__main__":
    """
    #######################################################################
    # same-domain data
    signal_issues = DataLoader.load_df_compressed(Const.SIGNAL_ISSUES)
    wire_issues = DataLoader.load_df_compressed(Const.WIRE_ISSUES)

    firefox_issues = DataLoader.load_df_compressed(Const.FIREFOX_ISSUES)
    brave_issues = DataLoader.load_df_compressed(Const.BRAVE_ISSUES)

    nextcloud_issues  = DataLoader.load_df_compressed(Const.NEXTCLOUD_ISSUES)
    owncloud_issues = DataLoader.load_df_compressed(Const.OWNCLOUD_ISSUES)

    allinputs = [[signal_issues, wire_issues], [firefox_issues, brave_issues], [nextcloud_issues, owncloud_issues]]



    top_num_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    results_domain_name = ['privacy_security', 'web_browser', 'office suite']
    
    for j in range(3):
        print(results_domain_name[j])
        for i in top_num_list:
            results = all_results(allinputs, i)
            print(i, results[j][4][0])

    for j in range(3):
        print(results_domain_name[j])
        results = all_results(allinputs, 50)
        print(results[j][4][1], results[j][5][0], results[j][5][1], results[j][5][2], results[j][5][3])
    """

    #signal_counter, wire_counter, signal_title_text, wire_title_text, results, plot_data = statistic(signal_issues, wire_issues, 50)
    #print(results[1], plot_data[0], plot_data[1], plot_data[2], plot_data[3])

    """
    signalcloud = WordCloud().generate(signal_title_text)
    plt.imshow(signalcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('signalclould.pdf')
    plt.show()

    wirecloud = WordCloud().generate(wire_title_text)
    plt.imshow(wirecloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wireclould.pdf')
    plt.show()
    """

    #print(results[0])

    """

    top_num_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    for i in top_num_list:
        sc, wic, stt, wtt, results_toprint, pd = statistic(signal_issues, wire_issues, i)
        print(i, results_toprint[0])
    """
    #######################################################################
    # different-domain data
    # A_random_index = np.random.permutation(dataset_length)

    A_issues = DataLoader.load_df_compressed(Const.OWNCLOUD_ISSUES)
    A_random_index = np.random.permutation(A_issues.shape[0])    
    A_issues = A_issues.iloc[A_random_index[:3000]]

    np.random.seed(2020)

    B_issues = DataLoader.load_df_compressed(Const.BRAVE_ISSUES)
    B_random_index = np.random.permutation(B_issues.shape[0]) 
    B_issues = B_issues.iloc[B_random_index[:3000]]

    top_num_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for i in top_num_list:
        sc, wic, stt, wtt, results_toprint, pd_ = statistic(A_issues, B_issues, i)
        print(i, results_toprint[0])

    
    results = statistic(A_issues, B_issues, 51)
    print(results[4][1], results[5][0], results[5][1], results[5][2], results[5][3])
