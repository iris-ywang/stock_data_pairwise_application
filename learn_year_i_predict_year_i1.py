import pandas as pd
import numpy as np
import os
import logging
from sklearn.svm import SVR, SVC
from load_data import GetData
from pairwise_approach import performance_standard_approach, performance_pairwise_approach
import multiprocessing


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":

    def run(year):
        data = GetData(os.getcwd() + "/stock_yearly_data/", year, prefix="stock_data")

        svc_params = {
            'C': [0.1, 0.5, 1, 2, 5, 10],
            'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
            'tol': [1e-3, 1e-2]
        }

        n_portofolio = 10
        ML_reg = SVR()
        ML_cls = SVC()
        return_sa = performance_standard_approach(data, ML_reg, n_portofolio, params = svc_params)
        returns_pa = performance_pairwise_approach(data, ML_cls, n_portofolio, params = svc_params)

        returns = return_sa + returns_pa
        logging.info(f"Year {year} predicted returns is \n {returns}")
        return (year, returns)
 
    with multiprocessing.Pool(processes=4) as executor:
        executor.map(run, range(2010, 2021), chunksize=1)