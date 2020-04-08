
import pandas as pd


class Utils:

    def load_set(self,path):
        return pd.read_csv(path)

    def features_target(self, dataset, drop_cols):

        X = dataset.drop(drop_cols, axis=1)

        return X


    def data_export(self,dataset,path):

        y = dataset.to_csv(path, index = False)

        return y










