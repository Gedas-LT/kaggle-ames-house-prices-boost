# Global variables for data_imputer function.
none_features = ["MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageCond", "GarageQual"]
zero_features = ["GarageYrBlt"]

# Global variables for ordinal_encoder function. 
quality_categories = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
quality_features = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageCond", "GarageQual", "HeatingQC"]