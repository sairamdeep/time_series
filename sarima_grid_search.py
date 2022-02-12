from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error as mse
import pickle
from pandas import read_csv
from numpy import array
import pandas as pd
import numpy as np
from tabulate import tabulate
import time,os
import warnings
warnings.filterwarnings('ignore')
from datetime import date, datetime
import collections
import yaml
import argparse
#data_dir = ''
#output_dir= ''

def yaml_load(path):
    with open(path) as f:
        Dict = yaml.safe_load(f)
    return Dict

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten2(d, parent_key='', sep='_'):
    sb={}
    for key in d:
        sb[key]=(flatten(d[key]))
    return sb


# SARIMA
def sarima_forecast(train,test, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(train, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(start=test.index[0], end=test.index[-1])
	return yhat

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return np.sqrt(mse(actual, predicted))

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred )/ y_true) * 100

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    #print('train split:',data[:-n_test].tail(3).index.tolist())
    #print('test split:',data[-n_test:].index.tolist())
    return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    predictions = sarima_forecast(train, test , cfg)
    # estimate prediction error
    #print(predictions.tolist())
    error = MAPE(test, predictions).tolist()
   
    rmse = np.sqrt(mse(test, predictions))#[mse(test[i], predictions[i]) for i in range(test.shape[0])]
    #print(error)
    #returning MAPE and RSME
    error.append(rmse)
    return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except Exception as ex:
            print(ex)
            result=[np.nan,np.nan,np.nan]
            error = None
    # check for an interesting result
    #if result is not None:
    #    print(' > Model[%s] %.3f' % (key, result))
    return_list=cfg
    return_list.extend(result)
    #print(return_list)
    return return_list

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    #scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    #scores.sort(key=lambda tup: tup[1])
    #print(scores)
    return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models


if __name__ == '__main__':
	# load dataset

    main_start = time.time()
    parser = argparse.ArgumentParser(description='Input config')
    parser.add_argument('-i', help='COnfig File Location', required=True)
    args = parser.parse_args()

    config_dict = yaml_load(args.i)
    
    data_dir = config_dict['data_dir']
    output_dir = config_dict['output_dir']
    ip_csv = config_dict['file_name']
    target=config_dict['target']
    start_date = config_dict['start_date']
    end_date = config_dict['end_date']
    n_test = config_dict['n_test']
    tag = config_dict['tag']
    df = read_csv(f'{data_dir}/{ip_csv}.csv')
    cred_list=df.PortfolioGroup.value_counts().index.tolist()
    #cred_list = ['BF 1']

    
    date_string = datetime.today().strftime('%Y_%m_%d')
    time_=datetime.today().strftime( "%H_%M")

    folder_string = date_string.replace('-','_')

    op_dir = f'{output_dir}/{folder_string}/{tag}'

    if os.path.exists(op_dir):
        print('Path exists:',op_dir)
    else:
        print('Creating folder:',op_dir)
        os.makedirs(op_dir)
    f_idfer=target+'_'+date_string
    e_list = []
    sub_dict = {}
    master_dict = {}
    cfg_cols= ['order', 'sorder', 'trend']
    target_cols=['M'+str(i) for i in range(1,n_test+1)]
    error_name = ['rmse']
    df_cols = []
    df_cols.extend(cfg_cols)
    df_cols.extend(target_cols)
    df_cols.extend(error_name)
    
    for i,cred in enumerate(cred_list[:]):
        print('---'*30)
        print('Portfolio : ',cred)
        try:
            start = time.time()
            #series=df[df.PortfolioGroup==cred_list[i]][target][:68]
            #history=df[df.PortfolioGroup==cred]#[target]
            history=df[df.PortfolioGroup==cred][df.year_month_date>=start_date][df.year_month_date<=end_date].set_index('year_month_date')[target]
            #history=history[start_idx:64]
            if history.shape[0]<2:
                e_list.append(cred)
                continue
           
            print('Training from :',history[:2].index.tolist())
            print('Testing from :',history[-n_test:].index.tolist())
            #continue
            # model configs
            cfg_list = sarima_configs(seasonal=[4,6,8,12])
            # grid search
            scores = grid_search(history, cfg_list, n_test)
            error_df=pd.DataFrame(scores,columns=df_cols)
          
            print('\n\t',cred,'\n')
            #print('target_cols:',target_cols)
            error_df['avg_error']=error_df[target_cols].mean(axis=1)
            error_df['median_error']=error_df[target_cols].median(axis=1)
            def get10Count(row):
                return np.sum(row<10)
            error_df['<10C']=error_df[target_cols].apply(get10Count,axis=1)
            f_name='SARIMA_OPS_GS_'+cred+'_'+f_idfer
            f_loc = f'{op_dir}/{f_name}.csv'
            print('Dumping file in',f_loc)
            error_df.to_csv(f_loc,index=False)

            #documenting
            
            train = history[:-n_test]
            test = history[-n_test:]
           

            test_dates = test.index.astype(str).tolist()
            actual_values = test.values.tolist()

            optimal_cfg = error_df.sort_values(by='rmse').head(1)[['order','sorder','trend']].values.tolist()
            optimal_cfg = [item for sublist in optimal_cfg for item in sublist]

            param_order = optimal_cfg[0]
            param_sorder = optimal_cfg[1]
            param_trend = optimal_cfg[2]
            train_start_date = train.index[:1].astype(str).tolist()[0]

            predictions=sarima_forecast(train,test, optimal_cfg)
            pred_dates = predictions.index.astype(str).tolist()
            pred_values =  predictions.values.tolist()

            assert set(test_dates) == set(pred_dates), "Damn! Test dates != Prediction Dates"

            actuals_dict = {k:v for k,v in zip(test_dates,actual_values)}
            preds_dict = {k:v for k,v in zip(pred_dates,pred_values)}
            mape_dict = {k:v for k,v in zip(pred_dates,MAPE(actual_values,pred_values))}

            sub_dict = {}
            sub_dict['train_start_date'] = train_start_date
            sub_dict['param_order'] = param_order
            sub_dict['param_sorder'] = param_sorder
            sub_dict['param_trend'] = param_trend
            sub_dict['Actual'] = actuals_dict
            sub_dict['Prediction'] = preds_dict
            sub_dict['MAPE'] = mape_dict

            master_dict[cred] = sub_dict

            error_df=error_df.sort_values(by=['rmse'],ascending=[True])
           
            print(tabulate(error_df.head(2), headers='keys', tablefmt='psql'))
            end = time.time()
            lapse=end-start
            print('----',f'Time for {cred} grid search :',round(lapse,2),'secs','----')
        except Exception as ex:
            print(ex)
            e_list.append(cred)
    #exit(0)
    pickle.dump(master_dict,open('actuals.srl','wb'))
    f_name='SARIMA_MASTER_'+f_idfer
    f_loc = f'{op_dir}/{f_name}.srl'
    pickle.dump(master_dict,open(f_loc,'wb'))
    f_loc = f'{op_dir}/{f_name}.csv'
    print('Dumping master file in',f_loc)
    m_df = pd.DataFrame.from_dict(flatten2(master_dict),orient='index')
    m_df.index.name = 'Portfolio'
    m_df.to_csv(f_loc,index=True)
    print('No Parameters for\n',e_list)

    end = time.time()
    lapse=end-main_start
    print('\n\n----',f'Time for complete grid search :',round(lapse,2),'secs','----')

