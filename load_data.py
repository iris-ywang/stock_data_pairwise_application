import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from itertools import permutations
from pa_basics.split_data import pair_test_with_train


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
        self.train_ary = self.train_df.to_numpy()
        self.test_ary = self.test_df.to_numpy()

        self.get_pair_test_pair_ids()


    def get_pair_test_pair_ids(self):
        self.train_test = np.concatenate((self.train_ary, self.test_ary), axis=0)
        self.train_ids = list(range(len(self.train_ary)))
        self.test_ids = list(range(len(self.train_ids), len(self.train_ids) + len(self.test_ary))) 
        self.train_pair_ids = list(permutations(self.train_ids, 2)) + [(a, a) for a in self.train_ids]
        self.c2_test_pair_ids = pair_test_with_train(self.train_ids, self.test_ids)
        self.c3_test_pair_ids = list(permutations(self.test_ids, 2)) + [(a, a) for a in self.test_ids]

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