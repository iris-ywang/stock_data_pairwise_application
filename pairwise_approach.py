import logging

import numpy as np
import logging

from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from pa_basics.all_pairs import pair_by_pair_id_per_feature
from pa_basics.rating import rating_trueskill
from load_data import GetData

import multiprocessing
import time

def build_ml_model(model, train_data, params=None, test_data=None):
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    n_cpus = multiprocessing.cpu_count()

    if params is not None:
        t = time.time()
        search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=4)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        logging.info(f"Training time: {time.time() - t}")
        logging.info(f"Best parameters: {search.best_params_}")
    else:
        logging.info("Not tuning")

        
    fitted_model = model.fit(x_train, y_train)

    if type(test_data) == np.ndarray:
        x_test = test_data[:, 1:]
        y_test_pred = fitted_model.predict(x_test)
        return fitted_model, y_test_pred
    else:
        return fitted_model

def get_index_of_top_x_item_in_a_array(ary, x):
    return np.argpartition(ary, -x)[-x:]

def mean_of_top_x_in_an_array(ary, x):
    return np.mean(ary[get_index_of_top_x_item_in_a_array(ary, x)])

def mean_of_top_x_in_other_array(ary, x_index):
    return np.mean(ary[x_index])


def calculate_returns(y_test_pred, y_test_true, n_portofolio):
    pred_index_of_top_x = get_index_of_top_x_item_in_a_array(y_test_pred, n_portofolio)
    mean_y_top_x = mean_of_top_x_in_an_array(y_test_pred, n_portofolio)
    mean_y_true_top_x = mean_of_top_x_in_other_array(y_test_true, pred_index_of_top_x)
    mean_y_top_x_true = mean_of_top_x_in_an_array(y_test_true, n_portofolio)
    logging.info(f"Predicted mean return is {mean_y_top_x + 1}, \
                 \n predicted true return is {mean_y_true_top_x + 1}, \
                 \n true return is {mean_y_top_x_true + 1}")
    return mean_y_true_top_x + 1, mean_y_top_x_true + 1


def performance_standard_approach(data: GetData, ML_reg, n_portofolios, params=None):
    logging.info("Running performance_standard_approach")
    train_ary = data.train_ary
    test_ary = data.test_ary
    # Default version:
    _, y_SA_d = build_ml_model(ML_reg, train_ary, test_data=test_ary)
    _, y_SA = build_ml_model(ML_reg, train_ary, test_data=test_ary, params=params)

    pred_true_return_list = {n: [] for n in n_portofolios}
    for n_portofolio in n_portofolios:
        pred_return_d, true_return_d = calculate_returns(y_SA_d, test_ary[:, 0] , n_portofolio)
        pred_return, true_return = calculate_returns(y_SA, test_ary[:, 0] , n_portofolio)

        pred_true_return_list[n_portofolio].append([pred_return_d, true_return_d])
        pred_true_return_list[n_portofolio].append([pred_return, true_return])

    return pred_true_return_list


def performance_pairwise_approach(data: GetData, ML_cls, n_portofolios: list,
                                  batch_size=1000000, params=None):
    logging.info("Running performance_pairwise_approach")
    runs_of_estimators = len(data.train_pair_ids) // batch_size
    Y_pa_c1_sign = []

    train_pairs_batch = pair_by_pair_id_per_feature(data=data.train_test,
                                                    pair_ids=data.train_pair_ids)
    train_pairs_for_sign = np.array(train_pairs_batch)
    train_pairs_for_sign[:, 0] = 2*(train_pairs_for_sign[:, 0] >= 0) - 1  # for logistic reg, using binary signs.
    rfc = build_ml_model(ML_cls, train_pairs_for_sign, params=params)
    Y_pa_c1_sign += list(train_pairs_for_sign[:, 0])

    # train_pairs_for_abs = np.absolute(train_pairs_batch)
    # rfr = build_ml_model(ML_reg, train_pairs_for_abs)


    if runs_of_estimators >= 1:
        logging.WARNING("Pairwise dataset is to large and over the batch size. Just returning results from one batch")


    c2_test_pair_ids = data.c2_test_pair_ids
    number_test_batches = len(c2_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c2_sign = []
    # Y_pa_c2_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches + 1:
            c2_test_pair_id_batch = c2_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            c2_test_pair_id_batch = c2_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = pair_by_pair_id_per_feature(data=data.train_test,
                                                       pair_ids=c2_test_pair_id_batch)

        Y_pa_c2_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        # Y_pa_c2_dist += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        # Y_pa_c2_true += list(test_pairs_batch[:, 0])
        if (test_batch + 1) * batch_size >= len(c2_test_pair_ids): break

    c3_test_pair_ids = data.c3_test_pair_ids
    number_test_batches = len(c3_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c3_sign = []
    # Y_pa_c3_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches:
            c3_test_pair_id_batch = c3_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            c3_test_pair_id_batch = c3_test_pair_ids[test_batch * batch_size:]

        test_pairs_batch = pair_by_pair_id_per_feature(data=data.train_test,
                                                       pair_ids=c3_test_pair_id_batch)
        Y_pa_c3_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        # Y_pa_c3_true += list(test_pairs_batch[:, 0])


    pred_true_return_list = {n: [] for n in n_portofolios}
    # A ranking C2 only:
    logging.info("C2 only")
    y_ranking = rating_trueskill(Y_pa_c2_sign, data.c2_test_pair_ids, data.train_test[:, 0])[data.test_ids]
    for n_portofolio in n_portofolios:
        pred_return_c2, true_return_c2 = calculate_returns(y_ranking, data.test_ary[:, 0] , n_portofolio)
        pred_true_return_list[n_portofolio].append([pred_return_c2, true_return_c2])
    
    # A ranking C3 only:
    logging.info("C3 only")
    y_ranking_c3 = rating_trueskill(Y_pa_c3_sign, data.c3_test_pair_ids, data.train_test[:, 0])[data.test_ids]
    for n_portofolio in n_portofolios:
        pred_return_c3, true_return_c3 = calculate_returns(y_ranking_c3, data.test_ary[:, 0] , n_portofolio)
        pred_true_return_list[n_portofolio].append([pred_return_c3, true_return_c3])
                                 
    # A ranking C2+C3 :
    logging.info("C2+C3")
    y_ranking_c2c3 = rating_trueskill(list(np.concatenate([Y_pa_c2_sign, Y_pa_c3_sign])),
                                      data.c2_test_pair_ids + data.c3_test_pair_ids,
                                      data.train_test[:, 0])[data.test_ids]
    for n_portofolio in n_portofolios:
        pred_return_c2c3, true_return_c2c3 = calculate_returns(y_ranking_c2c3, data.test_ary[:, 0] , n_portofolio)
        pred_true_return_list[n_portofolio].append([pred_return_c2c3, true_return_c2c3])

    # A ranking C1+C2+C3 :
    logging.info("C1+C2+C3")
    y_ranking_c1c2c3 = rating_trueskill(list(np.concatenate([Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign])),
                                      data.train_pair_ids + data.c2_test_pair_ids + data.c3_test_pair_ids,
                                      data.train_test[:, 0])[data.test_ids]
    for n_portofolio in n_portofolios:
        pred_return_c1c2c3, true_return_c1c2c3 = calculate_returns(y_ranking_c1c2c3, data.test_ary[:, 0] , n_portofolio)
        pred_true_return_list[n_portofolio].append([pred_return_c1c2c3, true_return_c1c2c3])

    # TODO: Check MSE of pairwise prediction - Next time

    return pred_true_return_list


def estimate_y_from_final_ranking_and_absolute_Y(test_ids, ranking, y_true, Y_c2_sign_and_abs_predictions):
    final_estimate_of_y_and_delta_y = {test_id: [] for test_id in test_ids}
    for pair_id, values in Y_c2_sign_and_abs_predictions.items():
        test_a, test_b = pair_id
        sign_ab = np.sign(ranking[test_a] - ranking[test_b])

        if test_a in test_ids and test_b not in test_ids:
            final_estimate_of_y_and_delta_y[test_a].append(y_true[test_b] + (values[0] * sign_ab))
        elif test_b in test_ids and test_a not in test_ids:
            final_estimate_of_y_and_delta_y[test_b].append(y_true[test_a] - (values[0] * sign_ab))

    mean_estimates = [np.mean(estimate_list)
                      for test_id, estimate_list in final_estimate_of_y_and_delta_y.items()]
    return mean_estimates

