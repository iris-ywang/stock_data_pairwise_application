import numpy as np
import os
import logging
from sklearn.svm import SVC, SVR
from load_data import GetData
from pairwise_approach import performance_standard_approach, performance_pairwise_approach
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def combine_returns(return_sa, return_pa, n_portofolios: list):
    all_returns_dict = {}
    for n_portofolio in n_portofolios:
        return_sa_per_size = return_sa[n_portofolio]
        return_pa_per_size = return_pa[n_portofolio]
        return_sa_per_size_ary = list(np.array(return_sa_per_size)[:, 0])
        return_pa_per_size_ary = list(np.array(return_pa_per_size)[:, 0])
        true_return = [return_sa_per_size[0][1]]
        all_returns = true_return + return_sa_per_size_ary + return_pa_per_size_ary
        all_returns_dict[n_portofolio] = all_returns

    return all_returns_dict


def run(year, n_portofolios, random_state=None):
    data = GetData(os.getcwd() + "/stock_yearly_data/", year, prefix="stock_data")
    
    '''
    params_reg = {
    'C': [0.1, 1, 10, 25, 100], 
    'gamma': [1,0.1,0.01, 0.001],
    }

    params_cls = {
    'C': [0.1, 1, 10, 25, 100], 
    'gamma': [1,0.1,0.01, 0.001],
    }
    '''

    params_reg = {
    'C': (1e-1, 100.0, 'log-uniform'), 
    'gamma': (1e-3, 1.0, 'log-uniform'),
    }

    params_cls = {
    'C': (1e-1, 100.0, 'log-uniform'), 
    'gamma': (1e-3, 1.0, 'log-uniform'),
    }

    n_cpus = multiprocessing.cpu_count()


    ML_reg = SVR()
    ML_cls = SVC()
    return_sa = performance_standard_approach(data, ML_reg, n_portofolios, params=params_reg)
    returns_pa = performance_pairwise_approach(data, ML_cls, n_portofolios, params=params_cls)

    all_returns_dict = combine_returns(return_sa, returns_pa, n_portofolios)
    logging.info(f"Year {year} predicted returns is \n {all_returns_dict}")
    logging.info("\n")
    return all_returns_dict



if __name__ == "__main__":
    # with multiprocessing.Pool(processes=3) as executor:a
    #     results = executor.map(run, range(2010, 2021), chunksize=1)

    results = []
    n_portofolios = [10, 20, 30, 50, 75]  
    for year in range(2010, 2021):
        all_returns_dict = run(year, n_portofolios, )
        for n_p, returns in all_returns_dict.items():
            returns = [year, n_p, 0] + returns
            results.append(returns)
        np.save("results_run_20231203_bayessearch_svm_2010.npy", np.array(results))



# TODO: Different portofolio size/ML methods/subcategories
# TODO: Add the method to use PDR averaging method for the same estimation & check MSE of the predictions of tuned models

# A record: results_run_20231028.npy, results_run_20231029.npy, results_run_20231030.npy are random states of n_portofolio=10, RF, random_search without criteria params
# A record: results_run_20231030_n_p_larger.npy, n_portofolio=20, RF, random_search with criteria params.

# 03/11/2023, running RF grid search 30191410