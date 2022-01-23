import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score


# Predict Values from model and dataframe
def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['#QueryID','Docid'])])


traindf = pd.read_csv("train.tsv",sep='\t')
testdf = pd.read_csv("test.tsv",sep='\t')

# ORIGINAL CODE
seed=8
gss = GroupShuffleSplit(test_size=.5, n_splits=1,random_state=seed).split(traindf,groups=traindf['#QueryID'])

X_train_inds, X_test_inds = next(gss)

train_data = traindf.iloc[X_train_inds]
X_train = train_data.loc[:,~train_data.columns.isin(['#QueryID','Docid','Label'])]
y_train = train_data.loc[:, train_data.columns.isin(['Label'])]

groups = train_data.groupby('#QueryID').size().to_frame('size')['size'].to_numpy()

test_data = traindf.iloc[X_test_inds]

X_test = test_data.loc[:, ~test_data.columns.isin(['Label'])] #drop #QueryId and Docid too?
y_test = test_data.loc[:, test_data.columns.isin(['#QueryID','Docid','Label'])]


model = xgb.XGBRanker(  
    tree_method='hist',
    booster='gbtree',
    objective='rank:ndcg',
    random_state=42, 
    learning_rate=0.1,
    colsample_bytree=0.9, 
    eta=0.05, 
    max_depth=6, 
    n_estimators=110, 
    subsample=0.75 
    )

model.fit(X_train, y_train, group=groups, verbose=True)


predictions = (X_test.groupby(['#QueryID','Docid'])
               .apply(lambda x: predict(model, x)))

yp1 = predictions.reset_index()[0].apply(lambda x : x[0])
yt1 = y_test['Label']

#calculates interim NDCG score based on holdout data
score = ndcg_score(y_true=[np.asarray(yt1)],y_score=[np.asarray(yp1)])
print("Model Score on initial testing data:", score)


# Runs prediction model and outputs the predicted values for the test.tsv data
out_pred = (testdf.groupby(['#QueryID','Docid'])
               .apply(lambda x: predict(model, x)))
out_pred = pd.DataFrame(out_pred,columns=['Score'])
out_pred['Score'] = out_pred['Score'].apply(lambda x : x[0])
out_pred.to_csv("A2.tsv", sep="\t", header=False)