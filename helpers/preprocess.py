import pandas as pd 

def preprocess_data(df):
    """
    Preprocess the insurance claims data
    """
    df = df.copy()
    
    # Drop ID columns - they have no predictive value
    id_cols = ['policy_id']  # Add any other ID columns here
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)
    
    # Parse max_torque: "250Nm@2750rpm" -> extract 250
    df['max_torque_nm'] = df['max_torque'].str.extract(r'(\d+\.?\d*)Nm').astype(float)
    df['max_torque_rpm'] = df['max_torque'].str.extract(r'@(\d+)rpm').astype(float)
    
    # Parse max_power: "87.8bhp@6000rpm" -> extract 87.8
    df['max_power_bhp'] = df['max_power'].str.extract(r'(\d+\.?\d*)bhp').astype(float)
    df['max_power_rpm'] = df['max_power'].str.extract(r'@(\d+)rpm').astype(float)
    
    df.drop(['max_torque', 'max_power'], axis=1, inplace=True)
    
    # Convert Yes/No to 1/0
    binary_cols = [col for col in df.columns if df[col].dtype == 'object' 
                   and set(df[col].unique()).issubset({'Yes', 'No'})]
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Only one-hot encode LOW cardinality categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    low_cardinality = [col for col in cat_cols if df[col].nunique() <= 20]
    high_cardinality = [col for col in cat_cols if df[col].nunique() > 20]
    
    if high_cardinality:
        print(f"Dropping high-cardinality columns: {high_cardinality}")
        print(f"  (Unique values: {[df[col].nunique() for col in high_cardinality]})")
        df.drop(columns=high_cardinality, inplace=True)
    
    # One-hot encode only low-cardinality columns
    if low_cardinality:
        df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)
    
    return df