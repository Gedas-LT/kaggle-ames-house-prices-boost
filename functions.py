import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from global_ import none_features, zero_features, quality_categories, quality_features

def data_imputer(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe, fills the missing values and returns pandas dataframe again.
    
    Also requires two global variables which are imported from separate file.
    """
    
    global none_features
    global zero_features
    
    column_imputer = ColumnTransformer(
    transformers=[
        ("constant_cat_inputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="None"), none_features),
        ("constant_num_inputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0), zero_features)
        ], 
    remainder=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    )
    
    array = column_imputer.fit_transform(input_df)

    features = []
    features.extend(none_features)
    features.extend(zero_features)

    for col in input_df:
        if col not in features:
            features.append(col)
    
    output_df = pd.DataFrame(array, columns=features)
    
    output_df = output_df.infer_objects()
    
    return output_df


def ordinal_encoder(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe and return pandas dataframe with encoded ordinal features.
    
    Also requires two global variables which are imported from separate file.
    """
    
    global quality_categories
    global quality_features
    
    def shape_transformer(categories: list, features: list) -> np.array:
        categories_arr = np.array([categories])
        repeated_categories_arr = np.repeat(categories_arr, len(features), axis=0)
        
        return repeated_categories_arr
    
    column_imputer = ColumnTransformer(
    transformers=[
        ("quality_ordinal_encoder", OrdinalEncoder(categories=list(shape_transformer(quality_categories, quality_features))), quality_features)
        ], 
    remainder="passthrough"
    )
    
    array = column_imputer.fit_transform(input_df)

    features = []
    features.extend(quality_features)

    for col in input_df:
        if col not in features:
            features.append(col)
    
    output_df = pd.DataFrame(array, columns=features)
    
    output_df = output_df.infer_objects()
    
    return output_df