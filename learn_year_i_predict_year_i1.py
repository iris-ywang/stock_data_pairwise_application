import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from load_data import GetData
from pairwise_approach import performance_standard_approach, performance_pairwise_approach
# import multiprocessing


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def run(year):
    data = GetData(os.getcwd() + "/stock_yearly_data/", year, prefix="stock_data")

    params = {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    n_portofolio = 10
    ML_reg = RandomForestRegressor(n_jobs=-1, random_state=11)
    ML_cls = RandomForestClassifier(n_jobs=-1, random_state=11)
    return_sa = performance_standard_approach(data, ML_reg, n_portofolio, params = params)
    returns_pa = performance_pairwise_approach(data, ML_cls, n_portofolio, params = params)

    returns = return_sa + returns_pa
    logging.info(f"Year {year} predicted returns is \n {returns}")
    logging.info("\n")
    return (year, returns)



if __name__ == "__main__":
    # with multiprocessing.Pool(processes=3) as executor:
    #     results = executor.map(run, range(2010, 2021), chunksize=1)
    results = []
    for year in range(2010, 2021):
        results.append(run(year))
        np.save("results_run_20231028.npy", np.array(results))



# TODO: Different portofolio size/ML methods/subcategories
# TODO: Add the method to use PDR averaging method for the same estimation & check MSE of the predictions of tuned models
