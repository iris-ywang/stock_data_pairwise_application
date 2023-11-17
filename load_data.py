import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from itertools import permutations
from pa_basics.split_data import pair_test_with_train


target_value_column_name = "annual_pc_price_change"

class GetData():
    def __init__(self, folder_path, learning_year, prefix="stock_data"):
        year1_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year}.csv", index_col=0)
        year2_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year+1}.csv", index_col=0)
        year3_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year+2}.csv", index_col=0)

        train_df = year1_df.merge(year2_df[target_value_column_name], how="left", left_index=True, right_index=True)
        train_df[target_value_column_name + '_x'] = train_df[target_value_column_name + '_y']
        train_df = train_df.drop([target_value_column_name + '_y'], axis=1).rename(columns={target_value_column_name + '_x': target_value_column_name}).reset_index()
        
        test_df = year2_df.merge(year3_df[target_value_column_name], how="left", left_index=True, right_index=True)
        test_df[target_value_column_name + '_x'] = test_df[target_value_column_name + '_y']
        test_df = test_df.drop([target_value_column_name + '_y'], axis=1).rename(columns={target_value_column_name + '_x': target_value_column_name}).reset_index()

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
        df = df.set_index("index")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df)
        # imputer.set_output(transform="pandas")
        df_ary = imputer.transform(df)
        df_imputed = pd.DataFrame(df_ary, columns=df.columns, index=df.index)
        assert (df_imputed[target_value_column_name]-df[target_value_column_name]).sum() == 0.0
        
        return df_imputed
        