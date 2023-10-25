import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer


target_value_column_name = "annual_pc_price_change"

class GetData():
    def __init__(self, folder_path, learning_year, prefix="stock_data"):
        train_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year}.csv", index_col=0)
        test_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year+1}.csv", index_col=0)

        self.train_df = train_df
        self.test_df = test_df
        self.process_train_and_test_data()
        self.train_company_index = self.train_df.index
        self.test_company_index = self.test_df.index


    def process_train_and_test_data(self):
        self.train_df = self.impute_missing_values_using_simple_imputer(
            self.remove_a_row_from_df_if_the_first_item_is_nan(
                self.train_df
            )
        )
        self.test_df = self.impute_missing_values_using_simple_imputer(
                    self.remove_a_row_from_df_if_the_first_item_is_nan(
                        self.test_df
                    )
                )

    @staticmethod
    def remove_a_row_from_df_if_the_first_item_is_nan(df):
        y_col_filtered = df[target_value_column_name].dropna(axis=0, how='any')
        df_filtered = df.loc[y_col_filtered.index]
        return df_filtered   
    
    @staticmethod
    def impute_missing_values_using_simple_imputer(df):
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df)
        imputer.set_output(transform="pandas")
        df_imputed = imputer.transform(df)
        return df_imputed


def build_ml_model(model, train_data, test_data=None):
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
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

    
if __name__ == "__main__":
    data = GetData(os.getcwd() + "/stock_yearly_data/", 2010, prefix="stock_data")

    n_portofolio = 10
    train_ary = data.train_df.to_numpy()
    test_ary = data.test_df.to_numpy()

    fitted_model, y_test_pred = build_ml_model(SVR(), train_ary, test_data=test_ary)
    index_of_top_x = get_index_of_top_x_item_in_a_array(y_test_pred, n_portofolio)
    mean_y_top_x = mean_of_top_x_in_an_array(y_test_pred, n_portofolio)
    mean_y_true_top_x = mean_of_top_x_in_other_array(test_ary[:,0], index_of_top_x)
    mean_y_top_x_true = mean_of_top_x_in_an_array(test_ary[:,0], n_portofolio)
    print(mean_y_top_x + 1, mean_y_true_top_x + 1, mean_y_top_x_true + 1)
    average_year_return = mean_y_top_x + 1
