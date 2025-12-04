import pandas as pd
from sklearn.preprocessing import OneHotEncoder
def brevage_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # date processing
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['year'] = df['Order_Date'].dt.year
    df['month'] = df['Order_Date'].dt.month
    df['day'] = df['Order_Date'].dt.day

    
    # one hot encoding for Region, Category, Product,Customer_Type
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_cols = ['Region', 'Category', 'Product', 'Customer_Type']
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=categorical_cols, inplace=True)
    
    # categorical to numerical for Customer_ID
    df['Customer_ID'] = df['Customer_ID'].astype('category').cat.codes
    
    # Dropping useless columns
    df.drop(columns=['Order_Date','Order_ID'], inplace=True)
    # Handle missing values by filling with 0
    df.fillna(0, inplace=True)
    
    return df

