import pandas as pd
import numpy as np
%matplotlib notebook
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv", encoding = "ISO-8859-1")
latlons = pd.read_csv('latlons.csv')
addresses = pd.read_csv('addresses.csv')

address = pd.merge(latlons, addresses, how = "inner", on = "address")

train_add = train_df.set_index('ticket_id').join(address.set_index('ticket_id'))
test_add = test_df.set_index('ticket_id').join(address.set_index('ticket_id'))
train_add = train_add[~train_add['hearing_date'].isnull()]
train_add = train_add[(train_add['compliance'] == 0) | (train_add['compliance'] == 1)]

train_remove_list = ['balance_due', 'collection_status', 'compliance_detail', 'payment_amount', 'payment_date', 'payment_status']
train_add.drop(train_remove_list, axis=1, inplace=True)

string_remove_list = ['violator_name', 'zip_code', 'country', 'city',
        'inspector_name', 'violation_street_number', 'violation_street_name',
        'violation_zip_code', 'violation_description', 'address',
        'mailing_address_str_number', 'mailing_address_str_name',
        'non_us_str_code', 'agency_name', 'state', 'disposition',
        'ticket_issued_date', 'hearing_date', 'grafitti_status', 'violation_code'
         ]

train_add.drop(string_remove_list, axis=1, inplace=True)
test_add.drop(string_remove_list, axis=1, inplace=True)

train_add.lat.fillna(method='pad', inplace=True)
train_add.lon.fillna(method='pad', inplace=True)
test_add.lat.fillna(method='pad', inplace=True)
test_add.lon.fillna(method='pad', inplace=True)

y_train = train_add.compliance
X_train = train_add.drop('compliance', axis=1)
X_test = test_add

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logit = LogisticRegression().fit(X_train, y_train)

y_score_logit = logit.decision_function(X_test)

probs = logit.predict_proba(X_test)[:,1]

test_final_df = pd.read_csv('test.csv', encoding = "ISO-8859-1")
test_final_df['compliance'] = probs
test_final_df.set_index('ticket_id', inplace=True)

print(test_final_df.compliance) 
