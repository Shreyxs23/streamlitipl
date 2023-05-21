#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')


# In[3]:


match.head()


# In[5]:


match.shape


# In[6]:


delivery.head(7)


# In[9]:


delivery.groupby(['match_id','inning']).sum()['total_runs']


# In[10]:


total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[11]:


total_score_df=total_score_df[total_score_df['inning']==1]


# In[12]:


total_score_df


# In[13]:


match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[14]:


match_df


# In[15]:


match_df['team1'].unique()


# In[18]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujrat Titans',
    'Lucknow Super Giants'
]


# In[20]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df['team1'] = match_df['team1'].str.replace('Gujrat Lions','Gujrat Titans')
match_df['team2'] = match_df['team2'].str.replace('Gujrat Lions','Gujrat Titans')

match_df['team1'] = match_df['team1'].str.replace('Pune Warriors','Lucknow Super Giants')
match_df['team2'] = match_df['team2'].str.replace('Pune Warriors','Lucknow Super Giants')


# In[21]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[22]:


match_df.shape


# In[23]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[24]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[25]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[26]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[27]:


delivery_df.head(20)


# In[28]:


delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[29]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[30]:


delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])


# In[31]:


delivery_df


# In[32]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10 - wickets
delivery_df.head()


# In[34]:


delivery_df.head()


# In[35]:


#crr=runs/over
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[36]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[37]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[38]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[39]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[40]:


final_df = final_df.sample(final_df.shape[0])


# In[41]:


final_df.sample()


# In[42]:


final_df.dropna(inplace=True)


# In[43]:


final_df = final_df[final_df['balls_left'] != 0]


# In[44]:


get_ipython().system('pip install sklearn')


# In[45]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[46]:


X_train


# In[68]:


get_ipython().system('pip install sklearn')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[70]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[71]:


pipe.fit(X_train,y_train)


# In[72]:


y_pred=pipe.predict(X_test)


# In[73]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[74]:


pipe.predict_proba(X_test)[10]


# In[75]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    


# In[76]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target


# In[77]:


temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[78]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[79]:


teams


# In[80]:


delivery_df['city'].unique()


# In[81]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




