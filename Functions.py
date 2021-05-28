import pandas as pd
import numpy as np
import random as rd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_pred(algo, df):
    preds = []
    for row in df.itertuples():
        prediction = algo.predict(getattr(row, 'userid'), getattr(row, 'itemid'))
        preds.append(prediction)
    df_pred = pd.DataFrame(preds)
    df_pred = df_pred.rename(index=str, columns={'uid': 'userid', 'iid': 'itemid', 'est': 'prediction'})
    df_pred.drop(['details', 'r_ui'], axis='columns', inplace=True)
    return df_pred


def get_all_pred(algo, user, items):
    preds = []
    for item in items:
        prediction = algo.predict(user, item)
        preds.append(prediction)
    df_pred = pd.DataFrame(preds)
    df_pred = df_pred.rename(index=str, columns={'uid': 'userid', 'iid': 'itemid', 'est': 'prediction'})
    df_pred.drop(['details', 'r_ui'], axis='columns', inplace=True)
    return df_pred


def merge_test_pred (test, pred):
    suff = ["_test", "_pred"]
    df_merged = pd.merge(test, pred, on = ['userid', 'itemid'], suffixes = suff)
    return df_merged[['rating', 'prediction']]


def get_metrics (comp, rating_sale):
    metrics = {}
    test = comp['rating']
    pred = comp['prediction']
    metrics['rmse'] = np.sqrt(mean_squared_error(test, pred))
    metrics['mae'] = mean_absolute_error(test, pred)
    comp_sale = comp[comp['rating'] == rating_sale]
    test_sale = comp_sale['rating']
    pred_sale = comp_sale['prediction']
    metrics['rmse sale'] = np.sqrt(mean_squared_error(test_sale, pred_sale))
    metrics['mae sale'] = mean_absolute_error(test_sale, pred_sale)    
    return metrics   


def get_sample_items (items, items_test, n):
    sample = rd.sample(items, k=n)
    diff = False
    
    while diff:
        if not(set(items_test) & set(sample)):
            diff = True
        else:
            sample = rd.sample(items, k=n)
    
    items_test = sample + items_test
    return items_test


def get_hit_rate (algo, df, k_list, rating_sale):
    k_dict = {}
    for k in k_list:
        k_dict[k] = {'hit': 0}
    df_sale = df[df['rating'] == rating_sale]
    users = list(df_sale['userid'].unique())
    items_list = list(df['itemid'].unique())
    total = len(users)
    for user in users:
        items_sales = list(df_sale[df_sale['userid'] == user]['itemid'].unique())
        samples = get_sample_items(items_list, items_sales, 200)
        predictions = get_all_pred(algo, user, samples)
        prediction = predictions.sort_values(by='prediction', ascending=False)
        for k in list(k_dict.keys()):
            top_pred = list(prediction['itemid'])[:k]
            for item in items_sales:
                if item in top_pred:
                    k_dict[k]['hit'] += 1
                    break
    
    for k1 in list(k_dict.keys()):
        k_dict[k1]['hit_rate'] = k_dict[k1]['hit'] / total
    
    return k_dict


def export_metrics (metrics, hit_rates, path):
    ks = list(hit_rates.keys())
    dict_metrics = {'RMSE': [metrics['rmse']],
                    'MAE': [metrics['mae']],
                    'RMSE Sale': [metrics['rmse sale']],
                    'MAE Sale': [metrics['mae sale']],
                    'Hit Rate K = ' + str(ks[0]): [hit_rates[ks[0]]['hit_rate']],
                    'Hit Rate K = ' + str(ks[1]): [hit_rates[ks[1]]['hit_rate']]}
    df_metrics = pd.DataFrame(data=dict_metrics)
    df_metrics.to_csv(path, index=False)