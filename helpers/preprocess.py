import pandas as pd 

def preprocess_data(df):
    """
    Preprocess the insurance claims data:
    1. Parse max_torque and max_power to extract numeric values
    2. Encode binary Yes/No columns
    3. One-hot encode categorical columns
    """
    df = df.copy()
    
    # Parse max_torque: "250Nm@2750rpm" -> extract 250
    df['max_torque_nm'] = df['max_torque'].str.extract(r'(\d+\.?\d*)Nm').astype(float)
    df['max_torque_rpm'] = df['max_torque'].str.extract(r'@(\d+)rpm').astype(float)
    
    # Parse max_power: "87.8bhp@6000rpm" -> extract 87.8
    df['max_power_bhp'] = df['max_power'].str.extract(r'(\d+\.?\d*)bhp').astype(float)
    df['max_power_rpm'] = df['max_power'].str.extract(r'@(\d+)rpm').astype(float)
    
    # Drop original columns
    df.drop(['max_torque', 'max_power'], axis=1, inplace=True)
    
    # Identify binary Yes/No columns
    binary_cols = [col for col in df.columns if df[col].dtype == 'object' 
                   and set(df[col].unique()).issubset({'Yes', 'No'})]
    
    # Convert Yes/No to 1/0
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Identify remaining categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df
