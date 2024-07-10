import json
import xgboost as xgb
import pandas as pd
import yaml
import pickle

class ModelScorer:
    """
    A class used to score models.

    ...

    Attributes
    ----------
    sql_config : dict
        Configuration for SQL queries.
    model_config : dict
        Configuration for the model.
    cat_cols : list
        List of categorical columns.
    num_cols : list
        List of numerical columns.
    xgb_model : XGBModel
        The XGBoost model.
    train_categories : dict
        Categories used during training.

    Methods
    -------
    replace_columns_with_suffix(df, suffix='_1'):
        Replaces columns in the DataFrame that have a matching column with the same name before the suffix.
    convert_bools_to_yes_no(df):
        Converts all True/False values in the DataFrame to 'yes'/'no'.
    align_categories(X_prod):
        Sets the categories of the production data to be the same as the ones used during training.
    model_predictor(df):
        Predicts the target variable using the model.
    """

    def __init__(self, sql_config_file='./sql_config.yml', model_config_file='./conf_telchurn.yml'):
        """
        Constructs all the necessary attributes for the ModelScorer object.

        Parameters
        ----------
            sql_config_file : str, optional
                Path to the SQL configuration file (default is 'sql_config.yml')
            model_config_file : str, optional
                Path to the model configuration file (default is 'conf_telchurn.yml')
        """
        with open(sql_config_file , 'r') as f:
            self.sql_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(model_config_file , 'r') as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        self.cat_cols = self.model_config['model']['features']['cat_features']
        self.num_cols = self.model_config['model']['features']['num_features']
        with open(self.model_config['model']['model_location'], 'rb') as f:
            self.xgb_model = pickle.load(f)
        with open(self.model_config['model']['train_category_levels'], 'r') as f:
            self.train_categories = json.load(f)

    def replace_columns_with_suffix(self, df: pd.DataFrame, suffix: str = '_1') -> pd.DataFrame:
        """
        Replaces columns in the DataFrame that have a matching column with the same name before the suffix.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        suffix : str, optional
            The suffix to look for in column names (default is '_1').

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with columns replaced.
        """
        suffixed_columns = [col for col in df.columns if col.endswith(suffix)]
        if not suffixed_columns:
            # If no suffixed columns are found, return the original DataFrame
            return df
        for col in suffixed_columns:
            original_col = col[:-len(suffix)]
            if original_col in df.columns:
                df[original_col] = df[col]
        df.drop(columns=suffixed_columns, inplace=True)
        return df

    def convert_bools_to_yes_no(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts all True/False values in the DataFrame to 'yes'/'no'.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with booleans converted to 'yes'/'no'.
        """
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype('category')
        df[bool_cols] = df[bool_cols].replace({True: 'yes', False: 'no'})
        return df

    def align_categories(self, X_prod):
        """
        Sets the categories of the production data to be the same as the ones used during training.

        Parameters
        ----------
        X_prod : pd.DataFrame
            The production data.

        Returns
        -------
        pd.DataFrame
            The production data with aligned categories.
        """
        for col in self.cat_cols:
            X_prod[col] = X_prod[col].astype('category').cat.set_categories(self.train_categories[col])
        return X_prod

    def model_predictor(self, df):
        """
        Predicts the target variable using the model.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with predictions.
        """
        df2 = self.replace_columns_with_suffix(df)
        df2 = self.convert_bools_to_yes_no(df2)
        for col in self.cat_cols:
            df2[col] = (df2[col]
                            .astype(str)
                            .replace('nan', 'unknown')
                            .str.lower()
                            .str.strip()
                            .replace('', 'unknown')
                            .fillna('unknown')
                            .astype('category'))

        for col in self.num_cols:
            df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)
        df3 = self.align_categories(df2)
        df3['new_prediction'] = self.xgb_model.predict(xgb.DMatrix(df3[self.xgb_model.feature_names], enable_categorical=True))
        return df3