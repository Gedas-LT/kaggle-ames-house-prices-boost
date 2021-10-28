import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures


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


def articial_features(input_df: pd.DataFrame, sqrt_features: list, area_features: list) -> pd.DataFrame:
    """Takes in pandas dataframe, two lists with name of features and 
    returns pandas dataframe with additional features.
    
     Also requires global variables which are imported from separate file.
    """
    
    # Age features.
    input_df["GarageAge"] = input_df["YrSold"] - input_df["GarageYrBlt"]
    input_df["HouseAge"] = input_df["YrSold"] - input_df["YearBuilt"]
    
    # SQRT features.
    for feature in sqrt_features:
        input_df[f"{feature}_Sqrt"] = np.sqrt(input_df[feature])
    
    # TotalHouseArea feature.
    input_df["TotalHouseArea"] = 0
    for feature in area_features:
        input_df["TotalHouseArea"] += input_df[feature]
        
    # Merged Area and Quality feature.
    input_df["AreaQuality"] = input_df["TotalHouseArea"] * input_df["OverallQual"]
    
    # Merged Area and Age feature.
    input_df["AreaAge"] = input_df["TotalHouseArea"] * input_df["HouseAge"]
    
    # Merged Quality and Condition feature.
    input_df["OverallQualandCond"] = input_df["OverallQual"] + input_df["OverallCond"]
    
    return input_df


def polynomial_features(input_df):
    """Takes in pandas dataframe and return pandas dataframe with 
    additional polynomial so called features.
    """
    
    numerical_features = [column for column in input_df if (input_df[column].dtypes == "int64") or (input_df[column].dtypes == "float64")]

    poly = PolynomialFeatures(2, include_bias=False)
    poly_array = poly.fit_transform(input_df[numerical_features])
    poly_columns = poly.get_feature_names_out(numerical_features)

    poly_data = pd.DataFrame(poly_array, columns=poly_columns)

    all_features = [column for column in input_df]
    poly_data = poly_data.drop(columns=all_features, errors="ignore")

    train_poly_data = input_df.join(poly_data)
    
    return train_poly_data