
# project_code_final: Analysis of the ELI Data #
## Ben Naismith ##

### This notebook ###

This notebook contains the most up-to-date and streamlined code for the project. For early cleaning and analysis efforts, please refer to the following documents:
- Cleaning: https://github.com/Data-Science-for-Linguists/Bigram-analysis-of-writing-from-the-ELI/tree/master/early_experiments/project_code1_cleaning.ipynb
- Analysis: https://github.com/Data-Science-for-Linguists/Bigram-analysis-of-writing-from-the-ELI/tree/master/early_experiments/project_code2_analysis.ipynb

### Table of contents ###

1.  [Data sharing plan](#1-Data-sharing-plan): description of sample data contents and licensing agreement
2.  [Initial setup](#2-Initial-setup): importing necessary modules
3.  [Student information](#3-Student-information): S_info_csv and S_info_df
4.  [Student responses](#4-Student-responses): answer_csv and answer_df
5.  [Course IDs](#5-Course-IDs): course_csv and course_df
6.  [User file internal](#6-user_file_internal): user_csv and user_df
7.  [Basic info about dataframes](#7-Basic-info-about-dataframes): description of dataframes in sections 3-6
8.  [Tokenization of answers](#8-Tokenization-of-answers): tokenization from answers_df
9.  [Bigrams](#9-Bigrams): creating bigram column from tokens
10. [Corpus frequency dictionary](#10-Corpus-frequency-dictionary): creating unigram frequency dictionary
11. [Bigram frequency dictionary](#11-Bigram-frequency-dictionary): creating bigram frequency dictionary
12. [Mutual Information](#12-Mutual-Information): creating a function for calculating MI
13. [Combo dataframe](#13-Combo-dataframe): combines earlier dataframes for easier analysis
14. [Occurrences per million](#14-Occurrences-per-million): create function to calculate occurrences per million
15. [bigram_df](#15-bigram_df): create new dataframe using new functions from earlier sections
16. [levels_df](#16-levels_df): create a small dataframe with useful overall statistics
17. [Pickling](#17-Pickling): saving pickles of dataframes and MI dict
18. [Visualizations](#18-Visualizations): create function to calculate occurrences per million

### 1. Data sharing plan ###

The full ELI data set (see project_plan.md) is private at this time. Below is a workbook with the current code for analyzing that data. In order to see how the code works, snippets of data have been displayed throughout.

A sample of the 'sanitized' data is included in the 'data' folder in this same repository. It contains samples of the four CSV files referred to in this code, consisting of 1000 answers, in order to allow for testing and reproducibility by others of the code. These 1000 answers are the first 1000 from the answer_csv file and correspond to user_file_id 7505 to 10108.

Ultimately, it is the intention of the dataset's authors for the entire dataset to be made public, with a CC license. Please see the LICENSE_notes.md for details

### 2. Initial setup ###


```python
#Import necesary modules
from __future__ import division
import numpy as np
import pandas as pd
import nltk
import glob
import matplotlib.pyplot as plt
import random

#return every shell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Create short-hand for directory root
cor_dir = "/Users/Benjamin's/Documents/ELI_Data_Mining/Data-Archive/1_sanitized/"
```


```python
#Add starter code created by Na-Rae Han for the ELI research group
from elitools import *
```

    Pretty printing has been turned OFF
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 48384 entries, 1 to 48420
    Data columns (total 8 columns):
    question_id        48384 non-null int64
    anon_id            48353 non-null object
    user_file_id       48384 non-null int64
    text               47175 non-null object
    directory          14 non-null object
    is_doublespaced    48384 non-null int64
    is_plagiarized     48384 non-null int64
    is_deleted         48384 non-null int64
    dtypes: int64(5), object(3)
    memory usage: 3.3+ MB
    <class 'pandas.core.frame.DataFrame'>
    Index: 913 entries, ez9 to bn6
    Data columns (total 20 columns):
    gender                       913 non-null object
    birth_year                   913 non-null int64
    native_language              913 non-null object
    language_used_at_home        912 non-null object
    language_used_at_home_now    855 non-null object
    non_native_language_1        859 non-null object
    yrs_of_study_lang1           863 non-null object
    study_in_classroom_lang1     863 non-null float64
    ways_of_study_lang1          863 non-null object
    non_native_language_2        309 non-null object
    yrs_of_study_lang2           312 non-null object
    study_in_classroom_lang2     863 non-null float64
    ways_of_study_lang2          863 non-null object
    non_native_language_3        55 non-null object
    yrs_of_study_lang3           59 non-null object
    study_in_classroom_lang3     863 non-null float64
    ways_of_study_lang3          863 non-null object
    createddate                  913 non-null object
    modifieddate                 909 non-null object
    course_history               913 non-null object
    dtypes: float64(3), int64(1), object(16)
    memory usage: 149.8+ KB


### 3. Student information
- S_info_csv
- S_info_df


```python
#Process the student_information.csv file
S_info_csv = cor_dir + "student_information.csv"
S_info_df = pd.read_csv(S_info_csv, index_col = 'anon_id')

S_info_df.head() #Issues still apparent with integers turned into floats
S_info_df.tail(10) #6 anon_id with no personal info - perhaps not students and to be 'pruned', as well as teachers with 'English' as the native language
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>birth_year</th>
      <th>native_language</th>
      <th>language_used_at_home</th>
      <th>language_used_at_home_now</th>
      <th>non_native_language_1</th>
      <th>yrs_of_study_lang1</th>
      <th>study_in_classroom_lang1</th>
      <th>ways_of_study_lang1</th>
      <th>non_native_language_2</th>
      <th>yrs_of_study_lang2</th>
      <th>study_in_classroom_lang2</th>
      <th>ways_of_study_lang2</th>
      <th>non_native_language_3</th>
      <th>yrs_of_study_lang3</th>
      <th>study_in_classroom_lang3</th>
      <th>ways_of_study_lang3</th>
      <th>createddate</th>
      <th>modifieddate</th>
      <th>course_history</th>
    </tr>
    <tr>
      <th>anon_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ez9</th>
      <td>Male</td>
      <td>1978</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>NaN</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Studied...</td>
      <td>Turkish</td>
      <td>less than 1 year</td>
      <td>0.0</td>
      <td>Studied by myself</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2006-01-30 15:07:18</td>
      <td>2006-03-14 15:13:37</td>
      <td>6;12;18;24;30</td>
    </tr>
    <tr>
      <th>gm3</th>
      <td>Male</td>
      <td>1980</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>NaN</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Had a native-speaker teacher;S...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2006-01-30 15:07:28</td>
      <td>2006-03-14 15:12:49</td>
      <td>6;12;24;30;38</td>
    </tr>
    <tr>
      <th>fg5</th>
      <td>Male</td>
      <td>1938</td>
      <td>Nepali</td>
      <td>Nepali</td>
      <td>NaN</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Had a n...</td>
      <td>French</td>
      <td>less than 1 year</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Had a n...</td>
      <td>Hindi</td>
      <td>more than 5 years</td>
      <td>0.0</td>
      <td>Studied by myself</td>
      <td>2006-01-30 15:07:45</td>
      <td>2006-03-14 15:11:36</td>
      <td>18;24</td>
    </tr>
    <tr>
      <th>ce5</th>
      <td>Female</td>
      <td>1984</td>
      <td>Korean</td>
      <td>Korean</td>
      <td>NaN</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Had a n...</td>
      <td>German</td>
      <td>1-2 years</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Listened to...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2006-01-30 15:07:49</td>
      <td>2006-03-14 15:12:24</td>
      <td>6;12;24;30;38;56</td>
    </tr>
    <tr>
      <th>fi7</th>
      <td>Female</td>
      <td>1982</td>
      <td>Korean</td>
      <td>Korean;Japanese</td>
      <td>NaN</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Had a native-speaker teacher;S...</td>
      <td>Japanese</td>
      <td>less than 1 year</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Listened to...</td>
      <td>French</td>
      <td>1-2 years</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Listened to...</td>
      <td>2006-01-30 15:07:52</td>
      <td>2006-03-14 15:12:17</td>
      <td>6;12;24;30;38</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>birth_year</th>
      <th>native_language</th>
      <th>language_used_at_home</th>
      <th>language_used_at_home_now</th>
      <th>non_native_language_1</th>
      <th>yrs_of_study_lang1</th>
      <th>study_in_classroom_lang1</th>
      <th>ways_of_study_lang1</th>
      <th>non_native_language_2</th>
      <th>yrs_of_study_lang2</th>
      <th>study_in_classroom_lang2</th>
      <th>ways_of_study_lang2</th>
      <th>non_native_language_3</th>
      <th>yrs_of_study_lang3</th>
      <th>study_in_classroom_lang3</th>
      <th>ways_of_study_lang3</th>
      <th>createddate</th>
      <th>modifieddate</th>
      <th>course_history</th>
    </tr>
    <tr>
      <th>anon_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ec5</th>
      <td>Female</td>
      <td>1963</td>
      <td>Chinese</td>
      <td>Chinese</td>
      <td>Chinese</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-06-16 14:08:05</td>
      <td>2011-06-16 14:13:03</td>
      <td>719;720;721;722;723;772;774;785;813;819;858;85...</td>
    </tr>
    <tr>
      <th>cy2</th>
      <td>Male</td>
      <td>1988</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>English</td>
      <td>less than 1 year</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Had a n...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:05</td>
      <td>2011-06-20 14:11:31</td>
      <td>845;846;847;871;872;927;928;931;949;950;1008;1...</td>
    </tr>
    <tr>
      <th>br9</th>
      <td>Female</td>
      <td>1981</td>
      <td>Chinese</td>
      <td>Chinese</td>
      <td>Chinese</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Studied...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:15</td>
      <td>2011-06-20 14:12:02</td>
      <td>868;869;870;871;872;947;951;953</td>
    </tr>
    <tr>
      <th>cl5</th>
      <td>Male</td>
      <td>1987</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>Arabic;English</td>
      <td>English</td>
      <td>less than 1 year</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Practiced s...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:23</td>
      <td>2011-06-20 14:13:16</td>
      <td>770;771;778;779;781;856;857;859;861;871;952;95...</td>
    </tr>
    <tr>
      <th>de1</th>
      <td>Male</td>
      <td>1983</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Teacher spo...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:27</td>
      <td>2011-06-20 14:12:02</td>
      <td>850;851;852;871;872;926;932;933;944;945;1008;1...</td>
    </tr>
    <tr>
      <th>ap0</th>
      <td>Male</td>
      <td>1978</td>
      <td>Japanese</td>
      <td>Japanese</td>
      <td>Japanese</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Listened to...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:33</td>
      <td>2011-06-20 14:12:52</td>
      <td>845;846;847;871;872</td>
    </tr>
    <tr>
      <th>gu4</th>
      <td>Male</td>
      <td>1983</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>Arabic;English</td>
      <td>Arabic</td>
      <td>more than 5 years</td>
      <td>0.0</td>
      <td>Studied by myself;I lived in a country where t...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:34</td>
      <td>2011-06-20 14:13:04</td>
      <td>772;773;774;775;776;868;869;870;871;872;922;92...</td>
    </tr>
    <tr>
      <th>hb0</th>
      <td>Female</td>
      <td>1980</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>Arabic</td>
      <td>English</td>
      <td>3-5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Had a native-speaker teacher;T...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:09:38</td>
      <td>2011-06-20 14:13:01</td>
      <td>851;869;870;871;872;923;942;944;945;946;1008;1...</td>
    </tr>
    <tr>
      <th>dp8</th>
      <td>Male</td>
      <td>1991</td>
      <td>Arabic</td>
      <td>Arabic;English</td>
      <td>Arabic;English</td>
      <td>English</td>
      <td>1-2 years</td>
      <td>1.0</td>
      <td>Studied grammar;Worked in pairs/groups;Had a n...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:10:15</td>
      <td>2011-06-20 14:13:57</td>
      <td>868;869;870;871;872</td>
    </tr>
    <tr>
      <th>bn6</th>
      <td>Male</td>
      <td>1986</td>
      <td>Arabic</td>
      <td>Arabic;English</td>
      <td>Arabic;English</td>
      <td>English</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Teacher spo...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2011-06-20 14:11:17</td>
      <td>2011-06-20 14:15:51</td>
      <td>860;861;862;871;872;930;947;948;949;951;998;99...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove anyone with 'English' or 'NaN' as their native_language, i.e. not students

#First try to create filters

Englishfilter = S_info_df['native_language'] == 'English' #first filter works
NaNfilter = S_info_df['native_language'] == np.nan #second filter doesn't

fake_Ss = S_info_df.loc[Englishfilter] #works, but...
fake_Ss

#fake_Ss = S_info_df.loc[(Englishfilter) or (NaNfilter)] #doesn't work
#fake_Ss

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>birth_year</th>
      <th>native_language</th>
      <th>language_used_at_home</th>
      <th>language_used_at_home_now</th>
      <th>non_native_language_1</th>
      <th>yrs_of_study_lang1</th>
      <th>study_in_classroom_lang1</th>
      <th>ways_of_study_lang1</th>
      <th>non_native_language_2</th>
      <th>yrs_of_study_lang2</th>
      <th>study_in_classroom_lang2</th>
      <th>ways_of_study_lang2</th>
      <th>non_native_language_3</th>
      <th>yrs_of_study_lang3</th>
      <th>study_in_classroom_lang3</th>
      <th>ways_of_study_lang3</th>
      <th>createddate</th>
      <th>modifieddate</th>
      <th>course_history</th>
    </tr>
    <tr>
      <th>anon_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ez7</th>
      <td>Male</td>
      <td>1987</td>
      <td>English</td>
      <td>Arabic</td>
      <td>Arabic;English</td>
      <td>Arabic</td>
      <td>more than 5 years</td>
      <td>0.0</td>
      <td>I lived in a country where they spoke Arabic</td>
      <td>English</td>
      <td>less than 1 year</td>
      <td>1.0</td>
      <td>Studied grammar;Studied vocabulary;Studied pro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2007-02-20 10:05:39</td>
      <td>2007-03-20 10:09:23</td>
      <td>156;167;180;191;200;212;223;234;245;256</td>
    </tr>
    <tr>
      <th>ay4</th>
      <td>Female</td>
      <td>1974</td>
      <td>English</td>
      <td>Korean</td>
      <td>Korean</td>
      <td>Korean</td>
      <td>more than 5 years</td>
      <td>1.0</td>
      <td>Studied grammar;Had a native-speaker teacher;S...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>other</td>
      <td>2009-06-09 12:04:22</td>
      <td>2009-11-13 12:43:36</td>
      <td>509;515;516;517;560;571;574;601;622;642;645</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Student responses###
- answer_csv
- answer_df


```python
#Process answer.csv file
answer_csv = cor_dir + "answer.csv"
answer_df = pd.read_csv(answer_csv, index_col = 'answer_id')

answer_df.head()
answer_df.tail(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>anon_id</th>
      <th>user_file_id</th>
      <th>text</th>
      <th>directory</th>
      <th>is_doublespaced</th>
      <th>is_plagiarized</th>
      <th>is_deleted</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>eq0</td>
      <td>7505</td>
      <td>I met my friend Nife while I was studying in a...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>am8</td>
      <td>7506</td>
      <td>Ten years ago, I met a women on the train betw...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>dk5</td>
      <td>7507</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>dk5</td>
      <td>7507</td>
      <td>I organized the instructions by time.</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>ad1</td>
      <td>7508</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>anon_id</th>
      <th>user_file_id</th>
      <th>text</th>
      <th>directory</th>
      <th>is_doublespaced</th>
      <th>is_plagiarized</th>
      <th>is_deleted</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48411</th>
      <td>6138</td>
      <td>dv8</td>
      <td>100847</td>
      <td>Early Second Language Education\r\r\r\nSaudi A...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48412</th>
      <td>6138</td>
      <td>ce1</td>
      <td>100848</td>
      <td>Publicly funded health care system\r\r\r\n\r\r...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48413</th>
      <td>6139</td>
      <td>fo7</td>
      <td>100911</td>
      <td>Happiness is the most effective feeling in peo...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48414</th>
      <td>6139</td>
      <td>fs9</td>
      <td>100912</td>
      <td>everyone want to play some games. some people ...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48415</th>
      <td>6139</td>
      <td>cl7</td>
      <td>100913</td>
      <td>Playing a game is fun only when you win?\r\r\r...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48416</th>
      <td>6139</td>
      <td>dr8</td>
      <td>100914</td>
      <td>Many people enjoy a game in their free time. B...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48417</th>
      <td>6137</td>
      <td>fv1</td>
      <td>100915</td>
      <td>\r\r\r\n                           ...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48418</th>
      <td>6137</td>
      <td>fo1</td>
      <td>100916</td>
      <td>Some  patients are suffering from the...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48419</th>
      <td>6119</td>
      <td>ge8</td>
      <td>100917</td>
      <td>My house looks amazing and modern. I decorated...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48420</th>
      <td>6027</td>
      <td>ge8</td>
      <td>100918</td>
      <td>History and Geography a...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 5. Course IDs ###
(to help with finding specific texts and linking other data frames)
- course_csv
- course_df


```python
#Process course.csv file
course_csv = cor_dir + "course.csv"
course_df = pd.read_csv(course_csv, index_col = 'course_id')

course_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_id</th>
      <th>level_id</th>
      <th>semester</th>
      <th>section</th>
      <th>course_description</th>
    </tr>
    <tr>
      <th>course_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>2064</td>
      <td>A</td>
      <td>Reading Pre_Intermediate 2064 A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>2064</td>
      <td>B</td>
      <td>Reading Low_Intermediate 2064 B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>2064</td>
      <td>M</td>
      <td>Reading Intermediate 2064 M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>2064</td>
      <td>P</td>
      <td>Reading Intermediate 2064 P</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>4</td>
      <td>2064</td>
      <td>Q</td>
      <td>Reading Intermediate 2064 Q</td>
    </tr>
  </tbody>
</table>
</div>



### 6. user_file_internal ###
- big csv file with a lot of information
- helps with finding specific texts and linking other data frames
- includes file_type_id, course_id, and many other fields


```python
#Process user_file_wavtxt.csv file
user_csv = cor_dir + "user_file_internal.csv"
user_df = pd.read_csv(user_csv, index_col = 'user_file_id')

user_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anon_id</th>
      <th>file_type_id</th>
      <th>file_info_id</th>
      <th>user_file_parent_id</th>
      <th>course_id</th>
      <th>session_id</th>
      <th>document_id</th>
      <th>activity</th>
      <th>order_num</th>
      <th>due_date</th>
      <th>...</th>
      <th>modifiedby</th>
      <th>modifieddate</th>
      <th>allow_submit_after_duedate</th>
      <th>allow_multiple_accesses</th>
      <th>allow_double_spacing</th>
      <th>duration</th>
      <th>pull_off_date</th>
      <th>direction</th>
      <th>grammar_qp_id</th>
      <th>is_deleted</th>
    </tr>
    <tr>
      <th>user_file_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>aj8</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>NaN</td>
      <td>2006-08-07 14:19:48</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fg8</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>NaN</td>
      <td>2006-08-07 14:19:48</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>be0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>NaN</td>
      <td>2006-08-07 14:19:48</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fc4</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>NaN</td>
      <td>2006-08-07 14:19:48</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fc4</td>
      <td>1</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>NaN</td>
      <td>2006-08-07 14:19:48</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



### 7. Basic info about dataframes ###

The following information is an overview of the four dataframes/csv files currently being looked at:

#### S_info_df ####
Size:
- there are 941 entries, i.e. students, although at least 9 need to be removed once filters can be made to work
- 21 columns including info about languages spoken, personal data like age, and learning preferences
- Some columns will likely be removed if deemed unhelpful/unnecessary (e.g. 4th language spoken)
- Some data is normalized, e.g. years of study, but others was open, resulting in very varied responses

Connection to other dataframes:
- link to answer_df is anon_id

Most useful columns for this project:
- anon_id (for linking to other df)
- L1, gender, time studying, age (for data analysis)  


#### answer_df ####
Size:
- there are 47175 'text' entries, i.e. student responses, although 48384 total rows. The remaining (including many null texts need to be removed as without texts they serve no purpose
- 9 columns including info about the question, the answer, and characteristics of the text (like if it was plagiarized)

Connection to other dataframes:
- link to S_info_df and course_df is anon_id column

Most useful columns for this project:
- answer_id (shorthand for the individual texts to be analyzed)
- text (the most important column so far) -> to be converted into tokens, bigrams, etc.  
- anon_id (for linking to other df)


#### course_df ####
Size:
- there are 1071 entries, i.e. one row for each course
- 6 columns including info about the course and class, both in terms of their assigned number and a description

Connection to other dataframes:
- link to user_df is course_id

Most useful columns for this project:
- only really useful as a transition for linking to other df  


#### user_df ####
Size:
- there are 76371 rows, each with a file_id number. However, it is unclear how to use this informatin effectively.
- There are 29 columns, although many are not useful for this project
- A lot of the cells have no input
- Some columns will likely be removed if deemed unhelpful/unnecessary

Connection to other dataframes:
- link to course_df is course_id column

Most useful columns for this project:
- course_id (to link to other DF)
- file_type_id (for indicating the type of activity used in class)


```python
S_info_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 913 entries, ez9 to bn6
    Data columns (total 20 columns):
    gender                       913 non-null object
    birth_year                   913 non-null int64
    native_language              913 non-null object
    language_used_at_home        912 non-null object
    language_used_at_home_now    855 non-null object
    non_native_language_1        859 non-null object
    yrs_of_study_lang1           863 non-null object
    study_in_classroom_lang1     863 non-null float64
    ways_of_study_lang1          863 non-null object
    non_native_language_2        309 non-null object
    yrs_of_study_lang2           312 non-null object
    study_in_classroom_lang2     863 non-null float64
    ways_of_study_lang2          863 non-null object
    non_native_language_3        55 non-null object
    yrs_of_study_lang3           59 non-null object
    study_in_classroom_lang3     863 non-null float64
    ways_of_study_lang3          863 non-null object
    createddate                  913 non-null object
    modifieddate                 909 non-null object
    course_history               913 non-null object
    dtypes: float64(3), int64(1), object(16)
    memory usage: 149.8+ KB



```python
answer_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 48384 entries, 1 to 48420
    Data columns (total 8 columns):
    question_id        48384 non-null int64
    anon_id            48353 non-null object
    user_file_id       48384 non-null int64
    text               47175 non-null object
    directory          14 non-null object
    is_doublespaced    48384 non-null int64
    is_plagiarized     48384 non-null int64
    is_deleted         48384 non-null int64
    dtypes: int64(5), object(3)
    memory usage: 3.3+ MB



```python
course_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1071 entries, 1 to 1123
    Data columns (total 5 columns):
    class_id              1071 non-null int64
    level_id              1071 non-null int64
    semester              1071 non-null int64
    section               1071 non-null object
    course_description    1058 non-null object
    dtypes: int64(3), object(2)
    memory usage: 50.2+ KB



```python
user_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 27134 entries, 1 to 100918
    Data columns (total 28 columns):
    anon_id                       26922 non-null object
    file_type_id                  27134 non-null int64
    file_info_id                  2151 non-null float64
    user_file_parent_id           25884 non-null float64
    course_id                     27134 non-null int64
    session_id                    26142 non-null float64
    document_id                   1599 non-null float64
    activity                      27134 non-null int64
    order_num                     2722 non-null float64
    due_date                      3286 non-null object
    post_date                     3714 non-null object
    assignment_name               2700 non-null object
    version                       27134 non-null int64
    directory                     0 non-null float64
    filename                      0 non-null float64
    content_text                  964 non-null object
    createdby                     24955 non-null object
    createddate                   27134 non-null object
    modifiedby                    462 non-null float64
    modifieddate                  462 non-null object
    allow_submit_after_duedate    27134 non-null int64
    allow_multiple_accesses       27134 non-null int64
    allow_double_spacing          27134 non-null int64
    duration                      406 non-null float64
    pull_off_date                 406 non-null object
    direction                     1997 non-null object
    grammar_qp_id                 0 non-null float64
    is_deleted                    27134 non-null int64
    dtypes: float64(10), int64(8), object(10)
    memory usage: 6.0+ MB


### 8. Tokenization of answers ###

Tokenizing the text in answer.csv to allow for further analysis, e.g., of bigrams.



```python
#column to tokenize
answer_df[['text']].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>I met my friend Nife while I was studying in a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ten years ago, I met a women on the train betw...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>In my country we usually don't use tea bags. F...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I organized the instructions by time.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creating 'toks' column and changing NaN to empty strings
answer_df = answer_df[answer_df['text'].notnull()]
answer_df['toks'] = answer_df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

answer_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>anon_id</th>
      <th>user_file_id</th>
      <th>text</th>
      <th>directory</th>
      <th>is_doublespaced</th>
      <th>is_plagiarized</th>
      <th>is_deleted</th>
      <th>toks</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>eq0</td>
      <td>7505</td>
      <td>I met my friend Nife while I was studying in a...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[I, met, my, friend, Nife, while, I, was, stud...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>am8</td>
      <td>7506</td>
      <td>Ten years ago, I met a women on the train betw...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[Ten, years, ago, ,, I, met, a, women, on, the...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>dk5</td>
      <td>7507</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>dk5</td>
      <td>7507</td>
      <td>I organized the instructions by time.</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[I, organized, the, instructions, by, time, .]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>ad1</td>
      <td>7508</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[First, ,, prepare, a, port, ,, loose, tea, ,,...</td>
    </tr>
  </tbody>
</table>
</div>



### 9. Bigrams###

Creating a bigram columns from the tok column



```python
#Creating a column of bigrams from the 'toks' column

answer_df['bigrams'] = answer_df.toks.apply(lambda x: list(nltk.bigrams(x)))
answer_df.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>anon_id</th>
      <th>user_file_id</th>
      <th>text</th>
      <th>directory</th>
      <th>is_doublespaced</th>
      <th>is_plagiarized</th>
      <th>is_deleted</th>
      <th>toks</th>
      <th>bigrams</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>eq0</td>
      <td>7505</td>
      <td>I met my friend Nife while I was studying in a...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[I, met, my, friend, Nife, while, I, was, stud...</td>
      <td>[(I, met), (met, my), (my, friend), (friend, N...</td>
    </tr>
  </tbody>
</table>
</div>



### 10. Corpus frequency dictionary ###

Create a frequency dictionary for all toks from answer_df


```python
#Joining all the answers before tokenizing them to create a corpus of tokens

answer_corpus = ' '.join(answer_df['text'])
answer_corpus[:100]
answer_corpus_tok = nltk.word_tokenize(answer_corpus)
answer_corpus_tok[:20]
```




    'I met my friend Nife while I was studying in a middle school. I was happy when I met him because he '






    ['I', 'met', 'my', 'friend', 'Nife', 'while', 'I', 'was', 'studying', 'in', 'a', 'middle', 'school', '.', 'I', 'was', 'happy', 'when', 'I', 'met']




```python
#Creating a dictionary from the answer_corpus

answer_dict = nltk.FreqDist(answer_corpus_tok)
random.sample(list(answer_dict.items()),5) #random 5-item sample
```




    [('friend=', 1), ('Vegetables', 23), ('hugged', 16), ('Booking', 3), ('disorders', 59)]



### 11. Bigram frequency dictionary ###
Create a bigram frequency dictionary from answer_corpus_tok


```python
#Creating bigrams from the answer_corpus_tok

answer_corpus_bigrams = list(nltk.bigrams(answer_corpus_tok))
answer_corpus_bigrams[:10]
```




    [('I', 'met'), ('met', 'my'), ('my', 'friend'), ('friend', 'Nife'), ('Nife', 'while'), ('while', 'I'), ('I', 'was'), ('was', 'studying'), ('studying', 'in'), ('in', 'a')]




```python
#Again creating a dictionary, this time a bigram dictionary

answer_bigram_dict = nltk.FreqDist(answer_corpus_bigrams)
random.sample(list(answer_bigram_dict.items()),5) #random 5-item sample
```




    [(('think', 'taking'), 2), (('coffee', 'when'), 4), (('Accent', 'coaches'), 1), (('unhappy', 'on'), 4), (('odor', '.'), 19)]



### 12. Mutual Information

Creating a function to calculate Mutual Information (MI), a useful measure of two-way collocation

(from https://corpus.byu.edu/mutualInformation.asp)  

Mutual Information is calculated as follows:  
MI = log ( (AB * sizeCorpus) / (A * B * span) ) / log (2)  

Suppose we are calculating the MI for the collocate color near purple in BYU-BNC.  

A = frequency of node word (e.g. purple): 1262  
B = frequency of collocate (e.g. color): 115  
AB = frequency of collocate near the node word (e.g. color near purple): 24  
sizeCorpus= size of corpus (# words; in this case the BNC): 96,263,399  
span = span of words (e.g. 3 to left and 3 to right of node word): 6  
log (2) is literally the log10 of the number 2: .30103  

MI = 11.37 = log ( (24 * 96,263,399) / (1262 * 115 * 6) ) / .30103  


```python
#The above formula turned into python code

import math
from math import log

def MI(word1, word2):
  prob_word1 = answer_dict[word1] / float(sum(answer_dict.values()))
  prob_word2 = answer_dict[word2] / float(sum(answer_dict.values()))
  prob_word1_word2 = answer_bigram_dict[word1, word2] / float(sum(answer_bigram_dict.values()))
  return math.log(prob_word1_word2/float(prob_word1*prob_word2),2)
```


```python
#Example of MI:

#This is a collocation which should have a medium strength MI (between 4-7)
answer_bigram_dict['young', 'people']
answer_dict['young']
answer_dict['people']

#'young' collocates strongly with 'people' (about 25% of time) but 'people' doesn't collocate strongly with 'young'
```




    469






    1605






    24516




```python
MI('young','people')

#That is the standard range for a M1 score
```




    5.840354713355728




```python
#Example #2 with two words that have a weaker MI

answer_bigram_dict['the', 'man']
answer_dict['the']
answer_dict['man']

MI('the', 'man')
```




    254






    171927






    1547






    2.1986947748534735



### 13. Combo dataframe
- joins answer_df, user_df, and course_df
- removes unnecessary columns
- narrows results down to only answers from writing classes and first versions of their work


```python
#join answer_df and user_df along 'user_file_id' column
combo_df = answer_df.join(user_df, on='user_file_id', lsuffix='user_file_id')

#now join this new df with course_df along 'course_id' column
combo_df = combo_df.join(course_df, on='course_id', lsuffix='user_file_id')
```


```python
#Dropping unnecessary columns (there are a lot)
combo_df = combo_df.drop(['directoryuser_file_id', 'is_doublespaced', 'is_plagiarized', 'is_deleteduser_file_id',
                            'modifiedby', 'modifieddate', 'allow_submit_after_duedate', 'anon_id', 'file_type_id',
                            'file_info_id', 'user_file_parent_id', 'createdby', 'session_id',
                           'document_id','filename', 'content_text', 'createddate', 'allow_multiple_accesses',
                           'directoryuser_file_id', 'is_doublespaced', 'is_plagiarized', 'is_deleteduser_file_id',
                           'modifiedby', 'modifieddate', 'allow_submit_after_duedate','activity', 'order_num',
                            'due_date', 'post_date', 'assignment_name', 'directory', 'activity', 'semester',
                            'order_num', 'due_date', 'post_date', 'assignment_name', 'allow_double_spacing',
                           'duration', 'pull_off_date', 'direction', 'grammar_qp_id', 'is_deleted',
                            'section', 'course_description'], axis = 1)
```


```python
#keeping only 1st versions of students' work
combo_df = combo_df.loc[combo_df['version'] == 1]

#'version' column now unnecessary
combo_df = combo_df.drop(['version'], axis = 1)
```


```python
#keeping only answers from writing classes (class_id = 2)
combo_df = combo_df.loc[combo_df['class_id'] == 2]

#'class_id' column now unnecessary
combo_df = combo_df.drop(['class_id'], axis = 1)
```


```python
#just change the order of columns to something more logical and rename some columns
combo_df = combo_df[['question_id','user_file_id', 'anon_iduser_file_id', 'level_id', 'course_id', 'text', 'toks', 'bigrams']]
combo_df.rename(columns={'anon_iduser_file_id':'anon_id'}, inplace=True)

#finished result =  much cleaner
combo_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>user_file_id</th>
      <th>anon_id</th>
      <th>level_id</th>
      <th>course_id</th>
      <th>text</th>
      <th>toks</th>
      <th>bigrams</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
      <td>[(In, my), (my, country), (country, we), (we, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>I organized the instructions by time.</td>
      <td>[I, organized, the, instructions, by, time, .]</td>
      <td>[(I, organized), (organized, the), (the, instr...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>[First, ,, prepare, a, port, ,, loose, tea, ,,...</td>
      <td>[(First, ,), (,, prepare), (prepare, a), (a, p...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>By time</td>
      <td>[By, time]</td>
      <td>[(By, time)]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>7509</td>
      <td>eg5</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare your cup, loose tea or bag tea,...</td>
      <td>[First, ,, prepare, your, cup, ,, loose, tea, ...</td>
      <td>[(First, ,), (,, prepare), (prepare, your), (y...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove level 2 (too few to be usefully analyzed)

combo_df.level_id.unique()

combo_df = combo_df.loc[combo_df['level_id'] != 2]

combo_df.level_id.unique()
```




    array([4, 5, 3, 2])






    array([4, 5, 3])




```python
#updated MI formula with combo_dict

def MI(word1, word2):
  prob_word1 = combo_unigram_dict[word1] / float(sum(combo_unigram_dict.values()))
  prob_word2 = combo_unigram_dict[word2] / float(sum(combo_unigram_dict.values()))
  prob_word1_word2 = combo_bigram_dict[word1, word2] / float(sum(combo_bigram_dict.values()))
  return math.log(prob_word1_word2/float(prob_word1*prob_word2),2)
```


```python
#create a column for total number of bigrams per text

combo_df['bigram_len'] = [len(x) for x in combo_df['bigrams']]
```


```python
#create a column of lowercase bigrams to use with MI
combo_df['bigrams_lower'] = [[(x.lower(), y.lower()) for x, y in element] for element in combo_df['bigrams']]
combo_df.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>user_file_id</th>
      <th>anon_id</th>
      <th>level_id</th>
      <th>course_id</th>
      <th>text</th>
      <th>toks</th>
      <th>bigrams</th>
      <th>bigram_len</th>
      <th>bigrams_lower</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
      <td>[(In, my), (my, country), (country, we), (we, ...</td>
      <td>67</td>
      <td>[(in, my), (my, country), (country, we), (we, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>I organized the instructions by time.</td>
      <td>[I, organized, the, instructions, by, time, .]</td>
      <td>[(I, organized), (organized, the), (the, instr...</td>
      <td>6</td>
      <td>[(i, organized), (organized, the), (the, instr...</td>
    </tr>
  </tbody>
</table>
</div>



#### Making the MI_sum column, i.e. the total MI of all the bigrams in each answer


```python
#create new freq dicts for combo_df (unigrams and bigrams) using same
#code as earlier versions with answer_df

combo_corpus = ' '.join(combo_df['text'])
combo_corpus_tok = nltk.word_tokenize(combo_corpus)
combo_corpus_tok = list(map(lambda x:x.lower(),combo_corpus_tok)) #making everything lowercase
combo_unigram_dict = nltk.FreqDist(combo_corpus_tok)

combo_corpus_bigrams = list(nltk.bigrams(combo_corpus_tok))
combo_bigram_dict = nltk.FreqDist(combo_corpus_bigrams)
```


```python
#updated MI formula with combo_dict and workaround to avoid math domain errors
def MI(word1, word2):
  prob_word1 = combo_unigram_dict[word1] / sum(combo_unigram_dict.values())
  prob_word2 = combo_unigram_dict[word2] / sum(combo_unigram_dict.values())
  prob_word1_word2 = combo_bigram_dict[word1, word2] / sum(combo_bigram_dict.values())
  y = prob_word1*prob_word2
  x = (prob_word1_word2/y) if y != 0 else 0
  if x != 0:
    return math.log(x,2)
  else:
    return 0
```


```python
#Create list of all text_MI scores (takes a while)

row = 0
text_MI = []

for x in combo_df['bigrams_lower']:
    y = [round(sum(MI(x[0], x[1]) for x in combo_df.iloc[row][9]),2)]
    row += 1
    text_MI.append(y)
```


```python
text_MI[:20] #check the results
```




    [[181.28], [7.65], [228.84], [-0.24], [120.11], [102.94], [269.31], [11.61], [229.97], [19.53], [160.22], [96.46], [188.73], [125.52], [74.17], [254.38], [190.67], [7.85], [407.79], [8.22]]




```python
len(combo_df['bigrams_lower'])
len(text_MI)
row
```




    12702






    12702






    12702




```python
text_MI = pd.Series(text_MI) #turn the list into a series
```


```python
#create a total of MI scores for each text (for machine learning later)
combo_df['MI_sum'] = [x[0] for x in text_MI]

combo_df.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>user_file_id</th>
      <th>anon_id</th>
      <th>level_id</th>
      <th>course_id</th>
      <th>text</th>
      <th>toks</th>
      <th>bigrams</th>
      <th>bigram_len</th>
      <th>bigrams_lower</th>
      <th>MI_sum</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
      <td>[(In, my), (my, country), (country, we), (we, ...</td>
      <td>67</td>
      <td>[(in, my), (my, country), (country, we), (we, ...</td>
      <td>181.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>I organized the instructions by time.</td>
      <td>[I, organized, the, instructions, by, time, .]</td>
      <td>[(I, organized), (organized, the), (the, instr...</td>
      <td>6</td>
      <td>[(i, organized), (organized, the), (the, instr...</td>
      <td>7.65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>[First, ,, prepare, a, port, ,, loose, tea, ,,...</td>
      <td>[(First, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>73</td>
      <td>[(first, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>228.84</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create an avg_bigram_MI scores for each text

combo_df['avg_bigram_MI'] = combo_df['MI_sum'] / combo_df['bigram_len']
```


```python
combo_df[['avg_bigram_MI']] = combo_df[['avg_bigram_MI']].apply(lambda x: pd.Series.round(x, 2)) #round to 2 decimals
combo_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>user_file_id</th>
      <th>anon_id</th>
      <th>level_id</th>
      <th>course_id</th>
      <th>text</th>
      <th>toks</th>
      <th>bigrams</th>
      <th>bigram_len</th>
      <th>bigrams_lower</th>
      <th>MI_sum</th>
      <th>avg_bigram_MI</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
      <td>[(In, my), (my, country), (country, we), (we, ...</td>
      <td>67</td>
      <td>[(in, my), (my, country), (country, we), (we, ...</td>
      <td>181.28</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>I organized the instructions by time.</td>
      <td>[I, organized, the, instructions, by, time, .]</td>
      <td>[(I, organized), (organized, the), (the, instr...</td>
      <td>6</td>
      <td>[(i, organized), (organized, the), (the, instr...</td>
      <td>7.65</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>[First, ,, prepare, a, port, ,, loose, tea, ,,...</td>
      <td>[(First, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>73</td>
      <td>[(first, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>228.84</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>By time</td>
      <td>[By, time]</td>
      <td>[(By, time)]</td>
      <td>1</td>
      <td>[(by, time)]</td>
      <td>-0.24</td>
      <td>-0.24</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>7509</td>
      <td>eg5</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare your cup, loose tea or bag tea,...</td>
      <td>[First, ,, prepare, your, cup, ,, loose, tea, ...</td>
      <td>[(First, ,), (,, prepare), (prepare, your), (y...</td>
      <td>49</td>
      <td>[(first, ,), (,, prepare), (prepare, your), (y...</td>
      <td>120.11</td>
      <td>2.45</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Let's also remove very short texts of less than 10 words which are not 'essays'

combo_df = combo_df.loc[combo_df['bigram_len'] >= 10]
combo_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>user_file_id</th>
      <th>anon_id</th>
      <th>level_id</th>
      <th>course_id</th>
      <th>text</th>
      <th>toks</th>
      <th>bigrams</th>
      <th>bigram_len</th>
      <th>bigrams_lower</th>
      <th>MI_sum</th>
      <th>avg_bigram_MI</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
      <td>[(In, my), (my, country), (country, we), (we, ...</td>
      <td>67</td>
      <td>[(in, my), (my, country), (country, we), (we, ...</td>
      <td>181.28</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>[First, ,, prepare, a, port, ,, loose, tea, ,,...</td>
      <td>[(First, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>73</td>
      <td>[(first, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>228.84</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>7509</td>
      <td>eg5</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare your cup, loose tea or bag tea,...</td>
      <td>[First, ,, prepare, your, cup, ,, loose, tea, ...</td>
      <td>[(First, ,), (,, prepare), (prepare, your), (y...</td>
      <td>49</td>
      <td>[(first, ,), (,, prepare), (prepare, your), (y...</td>
      <td>120.11</td>
      <td>2.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13</td>
      <td>7509</td>
      <td>eg5</td>
      <td>4</td>
      <td>115</td>
      <td>I organized the instructions by time, beacause...</td>
      <td>[I, organized, the, instructions, by, time, ,,...</td>
      <td>[(I, organized), (organized, the), (the, instr...</td>
      <td>38</td>
      <td>[(i, organized), (organized, the), (the, instr...</td>
      <td>102.94</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>7511</td>
      <td>fv6</td>
      <td>4</td>
      <td>115</td>
      <td>To make tea, nothing is easier, even if someti...</td>
      <td>[To, make, tea, ,, nothing, is, easier, ,, eve...</td>
      <td>[(To, make), (make, tea), (tea, ,), (,, nothin...</td>
      <td>98</td>
      <td>[(to, make), (make, tea), (tea, ,), (,, nothin...</td>
      <td>269.31</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>



### 14. Occurrences per million ###
- Create function for calculating occurrences per million  
- For unigrams and bigrams  

Formula:

FN = FO(1,000,000) / C

FN = normalized frequency
FO = observed frequency
C = corpus size


```python
#total number of unigrams
total_unigrams = len(combo_corpus_tok)

#total number of bigrams
total_bigrams = len(combo_corpus_bigrams)

total_unigrams
total_bigrams

#different by one a bigrams will be naturally be unigrams - 1 (for the first one)
```




    2549012






    2549011




```python
#create function where you enter the unigram and it tells you the frequency in the corpus per million tokens

def unigram_per_M(unigram):
   return (combo_unigram_dict[unigram]*1000000) / total_unigrams
```


```python
#create function where you enter the bigram and it tells you the frequency in the corpus per million tokens

def bigram_per_M(word1, word2):
   return (combo_bigram_dict[word1, word2]*1000000) / total_bigrams
```

### 15. bigram_df ###

Create bigram_df showing relevant info based on above formulas

- columns for this dataframe:
    - default index
    - bigrams
    - MI scores
    - occurrences per million
    - normalized percentage used at each proficiency level


```python
#Creating bigrams and tokens columns

bigram_df = pd.DataFrame.from_dict(combo_bigram_dict,orient='index')
bigram_df = bigram_df.reset_index()
bigram_df = bigram_df.rename(columns = {0:'tokens', 'index': 'bigram'})
bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(in, my)</td>
      <td>2629</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(my, country)</td>
      <td>875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(country, we)</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(we, usually)</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(usually, do)</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Changing bigram tuples to lists for easier manipulation

bigram_df['bigram'] = [list(x) for x in bigram_df['bigram']]
```

#### Creating MI column


```python
#Creating MI column (takes a few hours)

bigram_df['MI'] = [MI(x[0], x[1]) for x in bigram_df['bigram']]
```


```python
#Rounding results to two decimal places

bigram_df[['MI']] = bigram_df[['MI']].apply(lambda x: pd.Series.round(x, 2))
bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
      <th>MI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[in, my]</td>
      <td>2629</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[my, country]</td>
      <td>875</td>
      <td>5.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[country, we]</td>
      <td>17</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[we, usually]</td>
      <td>80</td>
      <td>3.41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[usually, do]</td>
      <td>53</td>
      <td>3.07</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating per_million column


```python
bigram_df['per_million'] = [bigram_per_M(x[0], x[1]) for x in bigram_df['bigram']]
```


```python
#Rounding to two decimal places

bigram_df[['per_million']] = bigram_df[['per_million']].apply(lambda x: pd.Series.round(x, 2))
bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
      <th>MI</th>
      <th>per_million</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[in, my]</td>
      <td>2629</td>
      <td>3.13</td>
      <td>1031.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[my, country]</td>
      <td>875</td>
      <td>5.50</td>
      <td>343.27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[country, we]</td>
      <td>17</td>
      <td>0.36</td>
      <td>6.67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[we, usually]</td>
      <td>80</td>
      <td>3.41</td>
      <td>31.38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[usually, do]</td>
      <td>53</td>
      <td>3.07</td>
      <td>20.79</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating 'normalized toks per level' and 'relative percentage per level' columns


```python
#create level dataframes
level_3 = combo_df.loc[combo_df['level_id'] == 3, :]
level_4 = combo_df.loc[combo_df['level_id'] == 4, :]
level_5 = combo_df.loc[combo_df['level_id'] == 5, :]

#create frequency dictionaries for each level
level_3_corpus = ' '.join(level_3['text'])
level_3_tok = nltk.word_tokenize(level_3_corpus)
level_3_tok = list(map(lambda x:x.lower(),level_3_tok))
level_3_bigrams = list(nltk.bigrams(level_3_tok))
level_3_bigram_dict = nltk.FreqDist(level_3_bigrams)

level_4_corpus = ' '.join(level_4['text'])
level_4_tok = nltk.word_tokenize(level_4_corpus)
level_4_tok = list(map(lambda x:x.lower(),level_4_tok))
level_4_bigrams = list(nltk.bigrams(level_4_tok))
level_4_bigram_dict = nltk.FreqDist(level_4_bigrams)

level_5_corpus = ' '.join(level_5['text'])
level_5_tok = nltk.word_tokenize(level_5_corpus)
level_5_tok = list(map(lambda x:x.lower(),level_5_tok))
level_5_bigrams = list(nltk.bigrams(level_5_tok))
level_5_bigram_dict = nltk.FreqDist(level_5_bigrams)
```


```python
#Example of what each cell should contain in the level_3 column
#level_3_bigram_dict divided by the value from combo_bigram_dict

#for example
"{0:.2f}%".format(level_3_bigram_dict['in', 'the'] / combo_bigram_dict['in', 'the'] * 100)

#totals for all 3 levels should add up to 100%
"{0:.2f}%".format(level_3_bigram_dict['in', 'the'] / combo_bigram_dict['in', 'the'] * 100)
"{0:.2f}%".format(level_4_bigram_dict['in', 'the'] / combo_bigram_dict['in', 'the'] * 100)
"{0:.2f}%".format(level_5_bigram_dict['in', 'the'] / combo_bigram_dict['in', 'the'] * 100)

12.17 + 40.75 + 47.07 #close enough!
```




    '12.06%'






    '12.06%'






    '40.50%'






    '47.01%'






    99.99000000000001




```python
#create updated freq dicts for combo_df (unigrams and bigrams)

combo_corpus = ' '.join(combo_df['text'])
combo_corpus_tok = nltk.word_tokenize(combo_corpus)
combo_corpus_tok = list(map(lambda x:x.lower(),combo_corpus_tok))
combo_unigram_dict = nltk.FreqDist(combo_corpus_tok)

combo_corpus_bigrams = list(nltk.bigrams(combo_corpus_tok))
combo_bigram_dict = nltk.FreqDist(combo_corpus_bigrams)
```


```python
#Checking that level bigram dicts add up to existing total bigram dict
level_3_bigram_dict['in', 'the']
level_4_bigram_dict['in', 'the']
level_5_bigram_dict['in', 'the']

level_3_bigram_dict['in', 'the'] + level_4_bigram_dict['in', 'the'] + level_5_bigram_dict['in', 'the']

combo_bigram_dict['in', 'the']
```




    1347






    4525






    5252






    11124






    11124




```python
#also necessary to normalize as different number of responses at each level

#weighting for each level
level_3_weighting = len(level_3.index) / len(combo_df.index)
level_4_weighting = len(level_4.index) / len(combo_df.index)
level_5_weighting = len(level_5.index) / len(combo_df.index)

level_3_weighting
level_4_weighting
level_5_weighting

level_3_weighting+level_4_weighting+level_5_weighting #should equal 100

#difference between observed and expected, i.e. expected weighting (.33) -  actual weighting (level_N_percent)
level_3_change = (1/3) - level_3_weighting
level_4_change = (1/3) - level_4_weighting
level_5_change = (1/3) - level_5_weighting

level_3_change
level_4_change
level_5_change

round(level_3_change + level_4_change + level_5_change, 2) # should be 0
```




    0.24625775830595106






    0.4115553121577218






    0.3421869295363271






    1.0






    0.08707557502738225






    -0.07822197882438847






    -0.008853596202993808






    -0.0




```python
#example of normalizing with ['in', 'the'] bigram

#un-normalized number
level_3_bigram_dict['in', 'the']
level_4_bigram_dict['in', 'the']
level_5_bigram_dict['in', 'the']
combo_bigram_dict['in', 'the']

#normalized number
n3 = level_3_bigram_dict['in', 'the'] + (combo_bigram_dict['in', 'the'] * level_3_change)
n4 = level_4_bigram_dict['in', 'the'] + (combo_bigram_dict['in', 'the'] * level_4_change)
n5 = level_5_bigram_dict['in', 'the'] + (combo_bigram_dict['in', 'the'] * level_5_change)

n3
n4
n5

n3 + n4 + n5
```




    1347






    4525






    5252






    11124






    2315.6286966046






    3654.8587075575024






    5153.512595837897






    11124.0




```python
#create a function for the above

def norm_toks_level3(word1, word2):
    return int((level_3_bigram_dict[word1,word2] + (combo_bigram_dict[word1,word2] * level_3_change)))

def norm_toks_level4(word1, word2):
    return int((level_4_bigram_dict[word1,word2] + (combo_bigram_dict[word1,word2] * level_4_change)))

def norm_toks_level5(word1, word2):
    return int((level_5_bigram_dict[word1,word2] + (combo_bigram_dict[word1,word2] * level_5_change)))

#Examples:
norm_toks_level3('in', 'the')
norm_toks_level4('in', 'the')
norm_toks_level5('in', 'the')
```




    2315






    3654






    5153




```python
#And as a comparative percentage
def norm_percent_level3(word1, word2):
    return(round(100*((level_3_bigram_dict[word1,word2] + (combo_bigram_dict[word1,word2] * level_3_change))
                      / (combo_bigram_dict[word1, word2]) if combo_bigram_dict[word1, word2] != 0 else 0),2))

def norm_percent_level4(word1, word2):
    return(round(100*((level_4_bigram_dict[word1,word2] + (combo_bigram_dict[word1,word2] * level_4_change))
                      / (combo_bigram_dict[word1, word2]) if combo_bigram_dict[word1, word2] != 0 else 0),2))

def norm_percent_level5(word1, word2):
    return(round(100*((level_5_bigram_dict[word1,word2] + (combo_bigram_dict[word1,word2] * level_5_change))
                      / (combo_bigram_dict[word1, word2]) if combo_bigram_dict[word1, word2] != 0 else 0),2))

#Examples:
norm_percent_level3('in', 'the')
norm_percent_level4('in', 'the')
norm_percent_level5('in', 'the')
```




    20.82






    32.86






    46.33




```python
#Normalized tokens pplied to the whole dataframe

bigram_df['lv3_norm_toks'] = [norm_toks_level3(x[0], x[1]) for x in bigram_df['bigram']]
bigram_df['lv4_norm_toks'] = [norm_toks_level4(x[0], x[1]) for x in bigram_df['bigram']]
bigram_df['lv5_norm_toks'] = [norm_toks_level5(x[0], x[1]) for x in bigram_df['bigram']]

bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
      <th>MI</th>
      <th>per_million</th>
      <th>lv3_norm_toks</th>
      <th>lv4_norm_toks</th>
      <th>lv5_norm_toks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[in, my]</td>
      <td>2629</td>
      <td>3.13</td>
      <td>1031.38</td>
      <td>687</td>
      <td>1124</td>
      <td>805</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[my, country]</td>
      <td>875</td>
      <td>5.50</td>
      <td>343.27</td>
      <td>249</td>
      <td>321</td>
      <td>298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[country, we]</td>
      <td>17</td>
      <td>0.36</td>
      <td>6.67</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[we, usually]</td>
      <td>80</td>
      <td>3.41</td>
      <td>31.38</td>
      <td>14</td>
      <td>51</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[usually, do]</td>
      <td>53</td>
      <td>3.07</td>
      <td>20.79</td>
      <td>6</td>
      <td>26</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
#And now the comparative percentages

bigram_df['lv3_rel_%'] = [norm_percent_level3(x[0], x[1]) for x in bigram_df['bigram']]
bigram_df['lv4_rel_%'] = [norm_percent_level4(x[0], x[1]) for x in bigram_df['bigram']]
bigram_df['lv5_rel_%'] = [norm_percent_level5(x[0], x[1]) for x in bigram_df['bigram']]

bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
      <th>MI</th>
      <th>per_million</th>
      <th>lv3_norm_toks</th>
      <th>lv4_norm_toks</th>
      <th>lv5_norm_toks</th>
      <th>lv3_rel_%</th>
      <th>lv4_rel_%</th>
      <th>lv5_rel_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[in, my]</td>
      <td>2629</td>
      <td>3.13</td>
      <td>1031.38</td>
      <td>687</td>
      <td>1124</td>
      <td>805</td>
      <td>26.28</td>
      <td>42.94</td>
      <td>30.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[my, country]</td>
      <td>875</td>
      <td>5.50</td>
      <td>343.27</td>
      <td>249</td>
      <td>321</td>
      <td>298</td>
      <td>28.71</td>
      <td>37.01</td>
      <td>34.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[country, we]</td>
      <td>17</td>
      <td>0.36</td>
      <td>6.67</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
      <td>14.59</td>
      <td>62.77</td>
      <td>22.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[we, usually]</td>
      <td>80</td>
      <td>3.41</td>
      <td>31.38</td>
      <td>14</td>
      <td>51</td>
      <td>13</td>
      <td>18.71</td>
      <td>64.68</td>
      <td>16.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[usually, do]</td>
      <td>53</td>
      <td>3.07</td>
      <td>20.79</td>
      <td>6</td>
      <td>26</td>
      <td>19</td>
      <td>12.48</td>
      <td>50.67</td>
      <td>36.85</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating level per_million columns


```python
#create per_million columns for each level

bigram_df['lv3_per_M'] = round(bigram_df['lv3_norm_toks']*1000000/total_bigrams, 2)
bigram_df['lv4_per_M'] = round(bigram_df['lv4_norm_toks']*1000000/total_bigrams, 2)
bigram_df['lv5_per_M'] = round(bigram_df['lv5_norm_toks']*1000000/total_bigrams, 2)
```


```python
bigram_df.index += 1 #frequency lists look better starting at 1

bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
      <th>MI</th>
      <th>per_million</th>
      <th>lv3_norm_toks</th>
      <th>lv4_norm_toks</th>
      <th>lv5_norm_toks</th>
      <th>lv3_rel_%</th>
      <th>lv4_rel_%</th>
      <th>lv5_rel_%</th>
      <th>lv3_per_M</th>
      <th>lv4_per_M</th>
      <th>lv5_per_M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>[in, my]</td>
      <td>2629</td>
      <td>3.13</td>
      <td>1031.38</td>
      <td>687</td>
      <td>1124</td>
      <td>805</td>
      <td>26.28</td>
      <td>42.94</td>
      <td>30.78</td>
      <td>269.52</td>
      <td>440.96</td>
      <td>315.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[my, country]</td>
      <td>875</td>
      <td>5.50</td>
      <td>343.27</td>
      <td>249</td>
      <td>321</td>
      <td>298</td>
      <td>28.71</td>
      <td>37.01</td>
      <td>34.29</td>
      <td>97.68</td>
      <td>125.93</td>
      <td>116.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[country, we]</td>
      <td>17</td>
      <td>0.36</td>
      <td>6.67</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
      <td>14.59</td>
      <td>62.77</td>
      <td>22.64</td>
      <td>0.78</td>
      <td>3.92</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[we, usually]</td>
      <td>80</td>
      <td>3.41</td>
      <td>31.38</td>
      <td>14</td>
      <td>51</td>
      <td>13</td>
      <td>18.71</td>
      <td>64.68</td>
      <td>16.61</td>
      <td>5.49</td>
      <td>20.01</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[usually, do]</td>
      <td>53</td>
      <td>3.07</td>
      <td>20.79</td>
      <td>6</td>
      <td>26</td>
      <td>19</td>
      <td>12.48</td>
      <td>50.67</td>
      <td>36.85</td>
      <td>2.35</td>
      <td>10.20</td>
      <td>7.45</td>
    </tr>
  </tbody>
</table>
</div>




```python
combo_df.head()
bigram_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>user_file_id</th>
      <th>anon_id</th>
      <th>level_id</th>
      <th>course_id</th>
      <th>text</th>
      <th>toks</th>
      <th>bigrams</th>
      <th>bigram_len</th>
      <th>bigrams_lower</th>
      <th>MI_sum</th>
      <th>avg_bigram_MI</th>
    </tr>
    <tr>
      <th>answer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>7507</td>
      <td>dk5</td>
      <td>4</td>
      <td>115</td>
      <td>In my country we usually don't use tea bags. F...</td>
      <td>[In, my, country, we, usually, do, n't, use, t...</td>
      <td>[(In, my), (my, country), (country, we), (we, ...</td>
      <td>67</td>
      <td>[(in, my), (my, country), (country, we), (we, ...</td>
      <td>181.28</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>7508</td>
      <td>ad1</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare a port, loose tea, and cup.\r\r...</td>
      <td>[First, ,, prepare, a, port, ,, loose, tea, ,,...</td>
      <td>[(First, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>73</td>
      <td>[(first, ,), (,, prepare), (prepare, a), (a, p...</td>
      <td>228.84</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>7509</td>
      <td>eg5</td>
      <td>4</td>
      <td>115</td>
      <td>First, prepare your cup, loose tea or bag tea,...</td>
      <td>[First, ,, prepare, your, cup, ,, loose, tea, ...</td>
      <td>[(First, ,), (,, prepare), (prepare, your), (y...</td>
      <td>49</td>
      <td>[(first, ,), (,, prepare), (prepare, your), (y...</td>
      <td>120.11</td>
      <td>2.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13</td>
      <td>7509</td>
      <td>eg5</td>
      <td>4</td>
      <td>115</td>
      <td>I organized the instructions by time, beacause...</td>
      <td>[I, organized, the, instructions, by, time, ,,...</td>
      <td>[(I, organized), (organized, the), (the, instr...</td>
      <td>38</td>
      <td>[(i, organized), (organized, the), (the, instr...</td>
      <td>102.94</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>7511</td>
      <td>fv6</td>
      <td>4</td>
      <td>115</td>
      <td>To make tea, nothing is easier, even if someti...</td>
      <td>[To, make, tea, ,, nothing, is, easier, ,, eve...</td>
      <td>[(To, make), (make, tea), (tea, ,), (,, nothin...</td>
      <td>98</td>
      <td>[(to, make), (make, tea), (tea, ,), (,, nothin...</td>
      <td>269.31</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bigram</th>
      <th>tokens</th>
      <th>MI</th>
      <th>per_million</th>
      <th>lv3_norm_toks</th>
      <th>lv4_norm_toks</th>
      <th>lv5_norm_toks</th>
      <th>lv3_rel_%</th>
      <th>lv4_rel_%</th>
      <th>lv5_rel_%</th>
      <th>lv3_per_M</th>
      <th>lv4_per_M</th>
      <th>lv5_per_M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>[in, my]</td>
      <td>2629</td>
      <td>3.13</td>
      <td>1031.38</td>
      <td>687</td>
      <td>1124</td>
      <td>805</td>
      <td>26.28</td>
      <td>42.94</td>
      <td>30.78</td>
      <td>269.52</td>
      <td>440.96</td>
      <td>315.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[my, country]</td>
      <td>875</td>
      <td>5.50</td>
      <td>343.27</td>
      <td>249</td>
      <td>321</td>
      <td>298</td>
      <td>28.71</td>
      <td>37.01</td>
      <td>34.29</td>
      <td>97.68</td>
      <td>125.93</td>
      <td>116.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[country, we]</td>
      <td>17</td>
      <td>0.36</td>
      <td>6.67</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
      <td>14.59</td>
      <td>62.77</td>
      <td>22.64</td>
      <td>0.78</td>
      <td>3.92</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[we, usually]</td>
      <td>80</td>
      <td>3.41</td>
      <td>31.38</td>
      <td>14</td>
      <td>51</td>
      <td>13</td>
      <td>18.71</td>
      <td>64.68</td>
      <td>16.61</td>
      <td>5.49</td>
      <td>20.01</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[usually, do]</td>
      <td>53</td>
      <td>3.07</td>
      <td>20.79</td>
      <td>6</td>
      <td>26</td>
      <td>19</td>
      <td>12.48</td>
      <td>50.67</td>
      <td>36.85</td>
      <td>2.35</td>
      <td>10.20</td>
      <td>7.45</td>
    </tr>
  </tbody>
</table>
</div>



### 16. levels_df ###

Create and overall numbers mini dataframe called levels_df


```python
#To see overall types and tokens by level

#first find length of sub-corpora
lv3_unigrams = len(level_3_tok)
lv4_unigrams = len(level_4_tok)
lv5_unigrams = len(level_5_tok)

lv3_bigrams = len(level_3_bigrams)
lv4_bigrams = len(level_4_bigrams)
lv5_bigrams = len(level_5_bigrams)

unigram_toks = pd.Series([lv3_unigrams, lv4_unigrams, lv5_unigrams, total_unigrams], index=['Level 3', 'Level 4', 'Level 5', 'Total'])
bigram_toks = pd.Series([lv3_bigrams, lv4_bigrams, lv5_bigrams, total_bigrams], index=['Level 3', 'Level 4', 'Level 5', 'Total'])
```


```python
#find number of types for each level and overall

total_unigram_types = len(set(combo_corpus_tok))
lv3_unigram_types = len(set(level_3_tok))
lv4_unigram_types = len(set(level_4_tok))
lv5_unigram_types = len(set(level_5_tok))

total_bigram_types = len(set(combo_corpus_bigrams))
lv3_bigram_types = len(set(level_3_bigrams))
lv4_bigram_types = len(set(level_4_bigrams))
lv5_bigram_types = len(set(level_5_bigrams))

unigram_types = pd.Series([lv3_unigram_types, lv4_unigram_types, lv5_unigram_types, total_unigram_types], index=['Level 3', 'Level 4', 'Level 5', 'Total'])
bigram_types = pd.Series([lv3_bigram_types, lv4_bigram_types, lv5_bigram_types, total_bigram_types], index=['Level 3', 'Level 4', 'Level 5', 'Total'])
```


```python
#find total number of texts at each level and overall

total_texts = len(combo_df.index)
lv3_texts = len(combo_df.loc[combo_df['level_id'] == 3, :])
lv4_texts = len(combo_df.loc[combo_df['level_id'] == 4, :])
lv5_texts = len(combo_df.loc[combo_df['level_id'] == 5, :])

texts = pd.Series([lv3_texts, lv4_texts, lv5_texts, total_texts], index=['Level 3', 'Level 4', 'Level 5', 'Total'])
```


```python
#create dataframe

levels_df = pd.concat([unigram_toks, unigram_types, bigram_toks, bigram_types, texts], axis = 1)
levels_df.columns = ['unigram_toks', 'unigram_types', 'bigram_toks', 'bigram_types', 'texts']
levels_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unigram_toks</th>
      <th>unigram_types</th>
      <th>bigram_toks</th>
      <th>bigram_types</th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Level 3</th>
      <td>282844</td>
      <td>11816</td>
      <td>282843</td>
      <td>81209</td>
      <td>2698</td>
    </tr>
    <tr>
      <th>Level 4</th>
      <td>1193172</td>
      <td>23231</td>
      <td>1193171</td>
      <td>236467</td>
      <td>4509</td>
    </tr>
    <tr>
      <th>Level 5</th>
      <td>1060753</td>
      <td>23667</td>
      <td>1060752</td>
      <td>236637</td>
      <td>3749</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>2549012</td>
      <td>39016</td>
      <td>2549011</td>
      <td>430738</td>
      <td>10956</td>
    </tr>
  </tbody>
</table>
</div>



### 17. Pickling ###

Saving pickles of dataframes and MI dict in order to save time in future and to use in other notebooks


```python
#save bigram_df as a pickle file and csv for later use

outfile = 'bigram_df.pkl'
bigram_df.to_pickle(outfile)
print(outfile, 'pickled.')

outfile = 'bigram_df.csv'
bigram_df.to_csv(outfile)
print(outfile, 'written out.')

#to read in later, use: pandas.read_pickle()
```

    bigram_df.pkl pickled.
    bigram_df.csv written out.



```python
#save combo_df as a pickle file and csv for later use

outfile = 'combo_df.pkl'
combo_df.to_pickle(outfile)
print(outfile, 'pickled.')

outfile = 'combo_df.csv'
combo_df.to_csv(outfile)
print(outfile, 'written out.')
```

    combo_df.pkl pickled.
    combo_df.csv written out.



```python
#save levels_df as a pickle file and csv for later use

outfile = 'levels_df.pkl'
levels_df.to_pickle(outfile)
print(outfile, 'pickled.')

outfile = 'levels_df.csv'
levels_df.to_csv(outfile)
print(outfile, 'written out.')
```

    levels_df.pkl pickled.
    levels_df.csv written out.



```python
#Make and pickle an MI_dict
import pickle

MI_dict = dict(zip(str(bigram_df.bigram), bigram_df.MI))

with open('MI_dict.pkl', 'wb') as handle:
    pickle.dump(MI_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('MI_dict.pkl written out.')
```

    MI_dict.pkl written out.


### 18. Visualizations

Visualizations based on this data can be found in a separate notebook:

https://github.com/Data-Science-for-Linguists/Bigram-analysis-of-writing-from-the-ELI/tree/master/Visualizations.ipynb
