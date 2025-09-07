import numpy as np
import pandas as pd

ext = '.csv'
start = 1991
end = 2021
ft = 'atp_matches_'

players = 'atp_players.csv'
rankings=['atp_rankings_20s.csv','atp_rankings_00s.csv','atp_rankings_10s.csv', 'atp_rankings_70s.csv','atp_rankings_80s.csv','atp_rankings_90s.csv',
'atp_rankings_current.csv']

def readfiles(ft,st,en):
    l = []
    for i in range(st,en+1):
        p = ft + str(i)+ ext
        print(p)
        df = pd.read_csv(p)
        l.append(df)
        #print(df.head(2))
    return l

files = readfiles(ft,start,end)
all_matches = pd.concat(files)
cat_cols = ['tourney_id','tourney_name','surface','draw_size','tourney_level','tourney_date','winner_id']

length = len(all_matches)/2
y= []
for i in range(int(length)):
    y.append(0)
for i in range(int(length)):
    y.append(1)
all_matches['y'] = pd.Series(y, index=all_matches.index)

all_matches1 = all_matches[:int(length)]
all_matches1.to_csv('all_matches_firsthalf.csv', encoding='utf-8', index=False)

all_matches2 = all_matches[int(length):]
all_matches2.to_csv('all_matches_secondhalf.csv', encoding='utf-8', index=False)

L  = list(all_matches1)
for i in range(17,27):
    L[i],L[i-10]=L[i-10],L[i]
all_matches1=all_matches1.reindex(columns=L)

all_matches1.rename (columns = {'loser_id' : 'Player1_id', 'loser_seed' : 'Player1_seed', 'loser_entry':
'Player1_entry','loser_name':'Player1_name','loser_hand':'Player1_hand','loser_ht':'Player1_ht','loser_ioc':
'Player1_ioc','loser_age':'Player1_age','loser_rank':'Player1_rank','loser_rank_points':'Player1_rank_points',
'winner_id':'Player2_id','winner_seed':'Player2_seed','winner_entry':'Player2_entry','winner_name':
'Player2_name','winner_hand':'Player2_hand','winner_ht':'Player2_ht','winner_ioc':'Player2_ioc',
'winner_age':'Player2_age','winner_rank':'Player2_rank,'winner_rank_points':'Player2_rank_points'},
inplace=True)

all_matches2.rename(columns={'loser_id':'Player2_id','loser_seed':'Player2_seed','loser_entry':
'Player2_entry','loser_name':'Player2_name','loser_hand':'Player2_hand','loser_ht':'Player2_ht', 'loser_ioc':'Player2_ioc','loser_age':'Player2_age','loser_rank':'Player2_rank','loser_rank_points':
'Player2_rank_points','winner_id':'Player1_id','winner_seed':'Player1_seed','winner_entry':'Player1_entry',
'winner_name':'Player1_name','winner_hand':'Player1_hand','winner_ht':'Player1_ht', 'winner_ioc':'Player1_ioc','winner_age':'Player1_age','winner_rank':'Player1_rank','winner_rank_points':
'Player1_rank_points'},inplace=True)

cc = [all_matches1,all_matches2]
all_matches_final= pd.concat(cc)

from sklearn.utils import shuffle

all_matches_final = shuffle(all_matches_final)
all_matches_final.to_csv('all_matches_final.csv', encoding='utf-8', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from scipy.special import legendre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns

df = pd.read_csv('/content/tennis_atp/all_matches_final.csv')

players = ['Player1_id','Player1_name','Player2_id','Player2_name']

to_drop = ['tourney_id','tourney_date','score']

cat_cols = ['tourney_name','surface','draw_size','tourney_level','Player1_entry','Player1_hand',
            'Player1_ioc','Player2_entry','Player2_hand','Player2_ioc','best_of','round']

num_cols = ['match_num','Player1_seed','Player1_ht','Player1_age','Player1_rank','Player1_rank_points',
            'Player2_seed','Player2_ht','Player2_age','Player2_rank','Player2_rank_points','minutes',
            'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced',
            'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']

df = df.drop(to_drop, axis=1)

for i in cat_cols:
    df[i] = df[i].replace(np.NaN, df[i].mode()[0])

for i in num_cols:
    print(i)
    df[i] = df[i].replace(np.NaN,df[i].mean())

def convertCatToNum(dff):
    dff_new = pd.get_dummies(dff, columns=cat_cols)
    return dff_new

df = convertCatToNum(df)

def normalize(dff,col_name_list):
    result = dff.copy()
    for feature_name in col_name_list:
        max_value = dff[feature_name].max()
        min_value = dff[feature_name].min()
        result[feature_name] = (dff[feature_name] - min_value) / (max_value - min_value)
    return result

df = normalize(df,num_cols)

num_bins = 10
for i in num_cols:
    plt.hist(df[i],num_bins,density=True, stacked= True,facecolor='green',alpha=0.4)
    plt.ylabel(i)
    plt.title("Data Distribution for " + i)
    plt.legend()
    plt.show()

sns.distplot(df[num_cols[1]], color='#aaff08', bins=100, kde_kws={"color": "k", "lw": 3, 
"label": num_cols[1]},  hist_kws={'alpha': 0.4});

sns.distplot(df[num_cols[0]], color='k', bins=100, kde_kws={"color": "r", "lw": 3, "label": num_cols[0]},
             hist_kws={'alpha': 0.2});
sns.distplot(df[num_cols[2]], color='y', bins=100, kde_kws={"color": "b", "lw": 3, "label": num_cols[2]},
             hist_kws={'alpha': 0.9});

sns.distplot(df[num_cols[3]], color='slategray', bins=100, kde_kws={"color": "k", "lw": 3, 
"label": num_cols[3]}, hist_kws={'alpha': 0.7});

sns.distplot(df[num_cols[4]], color='magenta', bins=100, kde_kws={"color": "k", "lw": 3,
 "label": num_cols[4]}, hist_kws={'alpha': 0.8});

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPRegressor

Y = pd.DataFrame(df['y'])
df = df.drop(['y'], axis=1)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train = X_train.drop(['Player1_id'], axis=1)
X_train = X_train.drop(['Player1_name'], axis=1)
X_train = X_train.drop(['Player2_id'], axis=1)
X_train = X_train.drop(['Player2_name'], axis=1)

test_names = []
for index, row in X_test.iterrows():
    v = []
    v.append(row['Player1_name'])
    v.append(row['Player2_name'])
    test_names.append(v)

X_test = X_test.drop(['Player1_id'], axis=1)
X_test = X_test.drop(['Player1_name'], axis=1)
X_test = X_test.drop(['Player2_id'], axis=1)
X_test = X_test.drop(['Player2_name'], axis=1)

from sklearn.linear_model import LogisticRegression

# Define model
model = LogisticRegression()
param_grid = {
    'C' : np.logspace(0, 4, num=10),
    'penalty' : ['l1', 'l2'],
    'solver' : ['liblinear', 'sag']
}
search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=10, scoring='accuracy', 
n_jobs=-1, random_state=1)
result = search.fit(X_train, y_train)
best_random = result.best_estimator_
y_pred_test = best_random.predict(X_test)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
print(classification_report(y_test, y_pred_test))
