import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from api_key_binance import API_SECURITY, API_KEY
from models.dl.hbnn import HBNN
from models.dl.lstm import LSTM
from models.dl.lstmd import LSTMD
from models.dl.tcn import TCN
from models.dl.lstm_gru_hybrid import LSTM_GRU
from models.dl.gru import GRU
from models.stats.arima import ARIMA
from models.stats.garch import GARCH
from models.ml.rf import RF
from models.ml.knn import KNN
from models.model_probabilistic import ModelProbabilistic
from models.dl.model_probabilistic_dl import ModelProbabilisticDL
from models.dl.model_interface_dl import ModelInterfaceDL
from models.ml.svr import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, explained_variance_score, mean_squared_log_error
from tensorflow import keras
from util import dataset, plot_training, save_results, dataset_binance
from darts.metrics import mape, mse, mae
from models.dl.tft import TFT
from models.ensembles.voting_regressor import VotingRegressor
from itertools import permutations, combinations

def Average(lst):
    return sum(lst) / len(lst)
def get_average_metrics(mses, rmses, maes, mapes, r2):
    avg_mae = Average(maes)
    avg_mse = Average(mses)
    avg_rmse = Average(rmses)
    avg_mape = Average(mapes)
    avg_r2 = Average(r2)
    r2.append(avg_r2)
    rmses.append(avg_rmse)
    mapes.append(avg_mape)
    mses.append(avg_mse)
    maes.append(avg_mae)


def r_squared(true, predicted):
    y = np.array(true)
    y_hat = np.array(predicted)
    
    y_m = y.mean()

    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y_m)**2)
    
    return 1 - (ss_res/ss_tot)

def get_all_possible_ensembles():
    iter_array= ['HYBRID', 'GRU', 'LSTM', 'TCN', 'TFT', 'RF', 'SVR', 'KNN', 'ARIMA']
    models = list()
    for i in range(1, (len(iter_array)+1)): 
        n_list = list(n for n in combinations(iter_array, i))
        for l in n_list: models.append(list(l))
    return models


def ensemble_hyperparam_test(models, clusters):
    for c in clusters:
        f_name = 'close-' + c + '-w30-h0_N'
        ensemble_name = '_'.join(models)
        print(f_name)
        en = VotingRegressor(name = ensemble_name, filename = f_name)
        en.models = models 
        en.hyperparametrization()
def train_ensemble_test(models, clusters, resources):
    for res in resources:               
        for c in clusters:
            mses, maes, rmses, mapes = [], [], [], []
            r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
            all_predictions, all_labels = [], []
            f_name = res + '-' + c + '-w30-h0'
            ensemble_name = '_'.join(models)
            experiment_name = 'train-' + ensemble_name + '-' + res + '-' + c  
            vr = VotingRegressor(name = ensemble_name, filename = f_name)
            vr.models = models
            predictions, true_values = vr.predict(weighted = False, train = True)
            print("MSE", mean_squared_error(true_values, predictions))
            print("MAE", mean_absolute_error(true_values, predictions))
            print("MAPE", mean_absolute_percentage_error(true_values, predictions))
            print("RMSE", np.sqrt(mean_squared_error(true_values, predictions)))
            rmses.append(np.sqrt(mean_squared_error(true_values, predictions)))
            mapes.append(mean_absolute_percentage_error(true_values, predictions))
            mses.append(mean_squared_error(true_values, predictions))
            maes.append(mean_absolute_error(true_values, predictions))
            r2_scores.append(r_squared(true_values, predictions))
            medaes.append(median_absolute_error(true_values, predictions))
            evs_scores.append(explained_variance_score(true_values, predictions))
            save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores, evs = evs_scores, medAE = medaes, rmsle = rmsles, msle = msles)
            save_results.save_output_csv(predictions, true_values, res, experiment_name, bivariate= False)

def train_ensemble_metrics(combine_models, res, clusters):
        for c in clusters:
            avg_maes, avg_mses, avg_rmses, avg_mapes, avg_r2 = [], [], [], [], []
            ensemble_models = []
            file_name =  'res/ensembles/metrics_train-all_ensembles' + '-' + res + '-' + c  +'.csv'
            for mod in combine_models:
                s = '_'.join(mod)
                path = 'res/ensembles/metrics_train-' + s + '-'+ res +'-' + c +'.csv'
                ensemble_models.append(s) 
                df = pd.read_csv(path)
                avg_maes.append(df['MAE'].iloc[0])
                avg_mses.append(df['MSE'].iloc[0])
                avg_rmses.append(df['RMSE'].iloc[0])
                avg_mapes.append(df['MAPE'].iloc[0])
                avg_r2.append(df['R2'].iloc[0])
            df_ensembles = pd.DataFrame(columns = ['ENSEMBLE_MODELS', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']) 
            df_ensembles['ENSEMBLE_MODELS'] = ensemble_models
            df_ensembles['MSE']= avg_mses
            df_ensembles['RMSE'] = avg_rmses
            df_ensembles['MAE'] = avg_maes
            df_ensembles['MAPE'] = avg_mapes
            df_ensembles['R2'] = avg_r2
            df_ensembles.to_csv(file_name)

def train_ranking_of_model(model, clusters, res):
    avg_mses_with, avg_mses_without= [], []
    file_name = 'res/ensembles/ranking-' + model.upper() + '-' + res + '_N.csv'
    for c in clusters:
        PATH =  'res/ensembles/metrics_train-all_ensembles' + '-' + res + '-' + c  +'_N.csv'
        df_ensembles = pd.read_csv(PATH)
        mses_with, mses_without = [], []
        for index, row in df_ensembles.iterrows():
            arr = row['ENSEMBLE_MODELS']
            if model != models: 
                models = arr.split('_')
                if model in models: mses_with.append(row['MSE'])
                else: mses_without.append(row['MSE'])
        avg_mses_with.append(Average(mses_with))
        avg_mses_without.append(Average(mses_without))
    df_ranking = pd.DataFrame(columns = ['CRYPTO', 'AVG_MSE_WITH', 'AVG_MSE_WITHOUT'])
    df_ranking['CRYPTO'] = clusters
    df_ranking['AVG_MSE_WITH'] = avg_mses_with
    df_ranking['AVG_MSE_WITHOUT'] = avg_mses_without
    df_ranking.to_csv(file_name)


def interval_ensemble_test(models, clusters, resources, timeframe = None):
    for res in resources:               
        for c in clusters:
            mses, maes, rmses, mapes = [], [], [], []
            r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
            all_predictions, all_labels = [], []
            for n in range(0, 10):  
                ensemble_name = '_'.join(models)
                if timeframe: 
                    f_name = res + '-' + c + '-w30-h0' + '-' + timeframe 
                    experiment_name = ensemble_name + '-' + res + '-' + c  + '-' + timeframe
                else: 
                    f_name = res + '-' + c + '-w30-h0' 
                    experiment_name = ensemble_name + '-' + res + '-' + c  
              
               
                vr = VotingRegressor(name = ensemble_name, filename = f_name)
                vr.models = models
                vr.path = 'res/outputs/output_'
                predictions, true_values = vr.predict(weighted = False, index = n)
                print("MSE", mean_squared_error(true_values, predictions))
                print("MAE", mean_absolute_error(true_values, predictions))
                print("MAPE", mean_absolute_percentage_error(true_values, predictions))
                print("RMSE", np.sqrt(mean_squared_error(true_values, predictions)))
                rmses.append(np.sqrt(mean_squared_error(true_values, predictions)))
                mapes.append(mean_absolute_percentage_error(true_values, predictions))
                mses.append(mean_squared_error(true_values, predictions))
                maes.append(mean_absolute_error(true_values, predictions))
                r2_scores.append(r_squared(true_values, predictions))
                medaes.append(median_absolute_error(true_values, predictions))
                evs_scores.append(explained_variance_score(true_values, predictions))
                all_predictions.append(predictions)
                all_labels.append(true_values)
            
            get_average_metrics(mses = mses, rmses =rmses, maes = maes, mapes = mapes, r2 = r2_scores)
            save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores)
            save_results.save_output_csv(all_predictions[-1],all_labels[-1], res, experiment_name,
                                                    bivariate=False)
            save_results.save_iteration_output_csv(preds= all_predictions, labels = all_labels, filename = experiment_name, iterations = 10)


def combine_ensemble_metrics(combine_models, res, clusters, timeframe =None):
        for c in clusters:
            avg_maes, avg_mses, avg_rmses, avg_mapes, avg_r2 = [], [], [], [], []
            ensemble_models = []
            if timeframe: 
                file_name =  'res/ensembles/metrics-all_ensembles' + '-' + res + '-' + c  + '-'+ timeframe +'_N.csv'
                eof = '-'+ timeframe +'_N.csv'
            else: 
                file_name =  'res/ensembles/metrics-all_ensembles' + '-' + res + '-' + c   +'_N.csv'
                eof = '_N.csv'
            
            for mod in combine_models:
                s = '_'.join(mod)
                path = 'res/ensembles/metrics_' + s + '-'+ res +'-' + c + eof
                ensemble_models.append(s) 
                df = pd.read_csv(path)
                avg_maes.append(df['MAE'].iloc[-1])
                avg_mses.append(df['MSE'].iloc[-1])
                avg_rmses.append(df['RMSE'].iloc[-1])
                avg_mapes.append(df['MAPE'].iloc[-1])
                avg_r2.append(df['R2'].iloc[-1])
            df_ensembles = pd.DataFrame(columns = ['ENSEMBLE_MODELS', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']) 
            df_ensembles['ENSEMBLE_MODELS'] = ensemble_models
            df_ensembles['MSE']= avg_mses
            df_ensembles['RMSE'] = avg_rmses
            df_ensembles['MAE'] = avg_maes
            df_ensembles['MAPE'] = avg_mapes
            df_ensembles['R2'] = avg_r2
            df_ensembles.to_csv(file_name)
def get_ranking_of_model(model, clusters, res, timeframe = None):
    avg_mses_with, avg_mses_without= [], []
    if timeframe:
        file_name = 'res/ensembles/ranking-' + model.upper() + '-' + res  +'-'+ timeframe +  '_N.csv'
        eof =  '-'+ timeframe +'_N.csv'
    else:
        file_name = 'res/ensembles/ranking-' + model.upper() + '-' + res  +  '_N.csv'
        eof =  '_N.csv'
    for c in clusters:
        PATH =  'res/ensembles/metrics-all_ensembles' + '-' + res + '-' + c  + eof
        df_ensembles = pd.read_csv(PATH)
        mses_with, mses_without = [], []
        for index, row in df_ensembles.iterrows():
            arr = row['ENSEMBLE_MODELS']
            models = arr.split('_')
            if model in models: mses_with.append(row['MSE'])
            else: mses_without.append(row['MSE'])
        avg_mses_with.append(Average(mses_with))
        avg_mses_without.append(Average(mses_without))
    df_ranking = pd.DataFrame(columns = ['CRYPTO', 'AVG_MSE_WITH', 'AVG_MSE_WITHOUT'])
    df_ranking['CRYPTO'] = clusters
    df_ranking['AVG_MSE_WITH'] = avg_mses_with
    df_ranking['AVG_MSE_WITHOUT'] = avg_mses_without
    df_ranking.to_csv(file_name)

def get_weighted_ensemble(models, weights, clusters, resources, timeframe =None):
    for res in resources:               
        for c in clusters:
            mses, maes, rmses, mapes = [], [], [], []
            r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
            all_predictions, all_labels = [], []
            ensemble_name = '_'.join(models)
            if timeframe:
                f_name = res + '-' + c + '-w30-h0' + '-'+ timeframe + '_N'
                experiment_name = ensemble_name + '-WEIGHTED-' + res + '-' + c  + '-'+ timeframe + '_N'
            else:
                f_name = res + '-' + c + '-w30-h0' + '_N'
                experiment_name = ensemble_name + '-WEIGHTED-' + res + '-' + c + '_N'
       
            
            vr = VotingRegressor(name = ensemble_name, filename = f_name)
            vr.models = models
            vr.path = 'res/ensembles/output_'
            vr.p['weights'] = weights
            predictions, true_values = vr.predict(weighted = True)
            print("MSE", mean_squared_error(true_values, predictions))
            print("MAE", mean_absolute_error(true_values, predictions))
            print("MAPE", mean_absolute_percentage_error(true_values, predictions))
            print("RMSE", np.sqrt(mean_squared_error(true_values, predictions)))
            rmses.append(np.sqrt(mean_squared_error(true_values, predictions)))
            mapes.append(mean_absolute_percentage_error(true_values, predictions))
            mses.append(mean_squared_error(true_values, predictions))
            maes.append(mean_absolute_error(true_values, predictions))
            r2_scores.append(r_squared(true_values, predictions))
            medaes.append(median_absolute_error(true_values, predictions))
            evs_scores.append(explained_variance_score(true_values, predictions))
            save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores)
            save_results.save_output_csv(predictions, true_values, res, experiment_name, bivariate= False)
            

def interval_weighted_ensemble_test(models, clusters, resources, weights, timeframe):
    for res in resources: 
        cluster_rmse, cluster_mse, cluster_mae, cluster_r2, cluster_mape = [], [], [], [], [] 
        ensemble_name = '_'.join(models)            
        c_experiment_name = ensemble_name + '-WEIGHTED-' + res + '-avg-' + timeframe +'_N'
           
        for c in clusters:
            mses, maes, rmses, mapes = [], [], [], []
            r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
            all_predictions, all_labels = [], []
                 
            for n in range(0, 10):
                f_name = res + '-' + c + '-w30-h0' + '-' + timeframe + '_N'
                
                experiment_name = ensemble_name + '-WEIGHTED-' + res + '-' + c  + '-' + timeframe +'_N'
                vr = VotingRegressor(name = ensemble_name, filename = f_name)
                vr.models = models
                vr.p['weights'] = weights
                vr.path = 'res/outputs/output_'
                predictions, true_values = vr.predict(weighted = True, index = n)
                """
                print("MSE", mean_squared_error(true_values, predictions))
                print("MAE", mean_absolute_error(true_values, predictions))
                print("MAPE", mean_absolute_percentage_error(true_values, predictions))
                print("RMSE", np.sqrt(mean_squared_error(true_values, predictions)))
                """
                rmses.append(np.sqrt(mean_squared_error(true_values, predictions)))
                mapes.append(mean_absolute_percentage_error(true_values, predictions))
                mses.append(mean_squared_error(true_values, predictions))
                maes.append(mean_absolute_error(true_values, predictions))
                r2_scores.append(r_squared(true_values, predictions))
                medaes.append(median_absolute_error(true_values, predictions))
                evs_scores.append(explained_variance_score(true_values, predictions))
                all_predictions.append(predictions)
                all_labels.append(true_values)
            get_average_metrics(mses = mses, rmses =rmses, maes = maes, mapes = mapes, r2 = r2_scores)
            save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores)
            save_results.save_iteration_output_csv(preds= all_predictions, labels = all_labels, filename = experiment_name, iterations = 10)
            cluster_rmse.append(rmses[-1])
            cluster_mse.append(mses[-1])
            cluster_r2.append(r2_scores[-1])
            cluster_mae.append(maes[-1])
            cluster_mape.append(mapes[-1])

        print('AVERAGE RMSE: ', Average(cluster_rmse))     
        get_average_metrics(mses = cluster_mse, rmses =cluster_rmse, maes = cluster_mae, mapes = cluster_mape, r2 = cluster_r2)
        save_results.save_metrics_csv(mses = cluster_mse, rmses =cluster_rmse, maes = cluster_mae, mapes = cluster_mape, filename = c_experiment_name, r2 =  cluster_r2)
               



periods = ['2m']
models = get_all_possible_ensembles()
"""
for m in models: 
  print(m)

#train_ensemble_test(models = ['HYBRID', 'GRU', 'LSTM', 'TCN', 'TFT'], cluster = ['btc'], resources =['close'])"
combine_ensemble_metrics(combine_models = models, res ='close', clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'], timeframe = '3m')
"""
interval_ensemble_test(models = ['LSTM', 'GRU'], clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'],  resources =['close'], timeframe = '1m')
all_models = ['HYBRID', 'LSTM', 'GRU', 'SVR', 'TCN', 'TFT', 'KNN', 'RF', 'ARIMA']
"""
for m in all_models:
    get_ranking_of_model(model = m, clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'], res ='close',  timeframe = '2m')

#get_weighted_ensemble(models = ['GRU', 'HYBRID', 'LSTM', 'TCN', 'ARIMA'], weights =(5, 5, 3, 1, 1), clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'], resources =['close'])
#get_weighted_ensemble(models = ['GRU', 'HYBRID'], weights =(2, 1), clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'], resources =['close'])

for t in periods:
    print(t)
    interval_weighted_ensemble_test(models = ['LSTM', 'HYBRID', 'GRU', 'TCN'] , clusters =  ['btc', 'ltc', 'eth', 'xmr', 'xrp'], resources = ['close'], weights = [9, 8 , 7, 6], timeframe = t)

for t in periods: 
  print(t)
  interval_ensemble_test(models = ['LSTM', 'GRU'], clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'],  resources =['close'], timeframe = t)
"""