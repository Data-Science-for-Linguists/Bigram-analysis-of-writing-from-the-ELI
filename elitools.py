
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import nltk
import glob
import matplotlib.pyplot as plt

get_ipython().magic('pprint #turn off pretty printing')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:

#Create short-hand for directory root
cor_dir = "../../ELI_Data_Mining/Data-Archive/1_sanitized/"

answer_df = pd.read_csv(cor_dir+'answer.csv', index_col = 'answer_id')  # answer_id as index. Unique! 


# In[3]:

answer_df.info()
answer_df.head()


# In[4]:

sinfo_df = pd.read_csv(cor_dir+'student_information.csv', index_col = 'anon_id')

sinfo_df.tail()


# In[5]:

sinfo_df.info()


# In[6]:

class_df = pd.read_csv(cor_dir + 'INDEX.class.csv')
class_df.head()

# Yep, doing this manually. It's small enough. 
class_type_map = {'r':1, 'w':2, 's':3, 'l':4, 'g':5}
class_type_map['r']
class_type_map2 = {1:'r', 2:'w', 3:'s', 4:'l', 5:'g'}


# In[7]:

course_df = pd.read_csv(cor_dir+'course.csv', index_col='course_id')
course_df.head()


# In[8]:

course_df['level_id'].value_counts()
course_df.loc[course_df['level_id']==1, :]


# In[9]:

user_file_internal_df = pd.read_csv(cor_dir+'user_file_internal.csv',  index_col='user_file_id')
user_file_internal_df.head()


# In[10]:

# Yep, am doing this... SHOULD REALLY BE DONE BY COLUMN. 
sinfo_df.fillna('', inplace=True)
answer_df.fillna('', inplace=True)


# In[11]:

# 188 Korean speakers
foo = sinfo_df.loc[sinfo_df['native_language']=="Korean", :]
len(foo)


# In[12]:

dir()


# In[13]:

# A bunch of helper functions below. 
# These are called from other functions. Not intended for end users. 

def cohist2courses(cohist):
    'Just a helper function'
    if cohist=='': return None
    courses = cohist.split(';')
    courses = [int(c) for c in courses]   # string -> integer
    return course_df.loc[courses, :]

def numsem(cohist):
    'Just a helper function'
    if cohist == '': return 0
    return len(cohist2courses(cohist)['semester'].unique())

def levelset(cohist):
    'Just a helper function'
    if cohist == '': return []
    return set(cohist2courses(cohist)['level_id'].unique())

def waslevel(levellist, cohist):
    'Just a helper function'
    return not set(levellist).isdisjoint(levelset(cohist)) 

numsem('6;12;18;24;30')
numsem('6;12;24;30;38;56')

levelset('6;12;24;30;38;56')
waslevel([1,3,4], '6;12;24;30;38;56')


# In[14]:

def get_user_courses(userid):
    '''Given anon_id, returns DF of all courses the user has taken
    '''
    course_hist = sinfo_df.loc[userid, 'course_history']  # '6;12;18;24;30'
    return cohist2courses(course_hist)

get_user_courses('ez9')
get_user_courses('ce5')


# In[15]:

l1_list = list(sinfo_df['native_language'].unique())
l1_list


# In[16]:

def user_filter(l1=l1_list, min_enroll=1, levels=[1,2,3,4,5]):
    '''Returns a list of anon_ids based on L1, minimum enrolled semesters, 
    and course levels he/she was ever in'''
    x = sinfo_df.loc[:, 'native_language'].isin(l1)
    y = sinfo_df.loc[:, 'course_history'].apply(numsem) >= min_enroll
    z = sinfo_df.loc[:, 'course_history'].apply(lambda ch: waslevel(levels, ch))
    return sinfo_df[x & y & z].index


# In[17]:

# Example usage. Arabic, Chinese or Korean speakers enrolled for at least 2 semesters
# and had ever taken an advanced level course. 
user_filter(l1=['Arabic', 'Chinese', 'Korean'], min_enroll=2, levels=[5])


# In[18]:

# Default setting, pulls 919 out of 920. One user missing. HUH! 
foo = set(user_filter())
allusers = set(sinfo_df.index)

allusers - foo


# In[19]:

# This user has no course history
#sinfo_df.loc['cm0']


# In[20]:

user_ar = user_filter(l1=['Arabic'])   # 288 of them
user_ch = user_filter(l1=['Chinese'])  # 173
user_ja = user_filter(l1=['Japanese']) # 59
user_ko = user_filter(l1=['Korean'])   # 188


# In[ ]:



