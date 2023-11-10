import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

    params_reg = {
    'criterion': ['mse'],
    'max_features': [1, 0.6, 0.3],
    'max_samples': [1, 0.8, 0.6, 0.4],
    'n_estimators': [100, 400, 800, 1200, 1600, 2000]
                     }
    # For pruning - we are doing it withou pruning
    # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    # 'min_samples_leaf': [1, 2, 4],
    # 'min_samples_split': [2, 5, 10],
    # Only available for scikit-learn 1.3+:
    # 'criterion': ['squared_error', 'friedman_mse', 'poisson'],
    # 'criterion': ['gini', 'entropy', 'log_loss'],

    params_cls = {
    'criterion': ['gini', 'entropy'],
    'max_features': [1, 0.6, 0.3],
    'max_samples': [1, 0.8, 0.6, 0.4],
    'n_estimators': [100, 400, 800, 1200, 1600, 2000]
    }

    n_cpus = multiprocessing.cpu_count()


    ML_reg = RandomForestRegressor(n_jobs=-1, random_state=random_state)
    ML_cls = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    return_sa = performance_standard_approach(data, ML_reg, n_portofolios, params=params_reg)
    returns_pa = performance_pairwise_approach(data, ML_cls, n_portofolios, params=params_cls)

    all_returns_dict = combine_returns(return_sa, returns_pa, n_portofolios)
    logging.info(f"Year {year} predicted returns is \n {all_returns_dict}")
    logging.info("\n")
    return all_returns_dict



if __name__ == "__main__":
    # with multiprocessing.Pool(processes=3) as executor:
    #     results = executor.map(run, range(2010, 2021), chunksize=1)

    results = []
    n_portofolios = [10, 20, 30, 50, 75]  
    for year in range(2010, 2021):
        for rs in [111,222,333]:
            if (year==2017) and (rs in [111,222]):
                logging.info(f"{year} with {rs} has been evaluated in the previous run. Skipping now. ")
                continue
            
            all_returns_dict = run(year, n_portofolios, random_state=rs)
            for n_p, returns in all_returns_dict.items():
                returns = [year, n_p, rs] + returns
                results.append(returns)
        np.save("results_run_20231103_gridsearch_rf.npy", np.array(results))



# TODO: Different portofolio size/ML methods/subcategories
# TODO: Add the method to use PDR averaging method for the same estimation & check MSE of the predictions of tuned models

# A record: results_run_20231028.npy, results_run_20231029.npy, results_run_20231030.npy are random states of n_portofolio=10, RF, random_search without criteria params
# A record: results_run_20231030_n_p_larger.npy, n_portofolio=20, RF, random_search with criteria params.

# 03/11/2023, running RF grid search 30191410