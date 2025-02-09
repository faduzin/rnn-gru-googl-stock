import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(df,
               remove_na=True,
               remove_duplicates=True,
               remove_outliers=True):
    if remove_na:
        try:
            df = df.dropna()
        except:
            print('Error removing NA values')

    if remove_duplicates:
        try:
            df = df.drop_duplicates()
        except:
            print('Error removing duplicates')

    if remove_outliers:
        try:
            num_rows_before = df.shape[0]
            for column in df.select_dtypes(include=['int', 'float']).columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            num_rows_after = df.shape[0]
            print(f'{num_rows_before - num_rows_after} outliers removed.')
        except:
            print('Error removing outliers')
    
    return df


def label_encode(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            print(f'Error encoding column {col} labels')
    
    return df


def data_normalize(df,
                   method='standard'):
    match method:
        case 'standard':
            scaler = StandardScaler()
        case 'minmax':
            scaler = MinMaxScaler()
        case _:
            print('Invalid normalization method')
            return df
        
    try:
        df = scaler.fit_transform(df)
        print('Data normalized successfully.')
    except:
        print('Error normalizing data.')

    return df


def preprocess_data(df,
                    remove_na=True,
                    remove_duplicates=True,
                    remove_outliers=True,
                    encode_labels=True):
    df = clean_data(df, remove_na, remove_duplicates, remove_outliers)
    
    if encode_labels:
        df = label_encode(df)
    
    return df


def split_data(X, y, test_size=0.2, shuffle=True):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
        print('Data split successfully.')
        return X_train, X_test, y_train, y_test
    except:
        print('Error splitting data.')
        return None
    

def plot_boxplot_and_histogram(df, column_name):
    
    fig, axes = plt.subplots(1, 
                             2,  
                             figsize=(12,4)) # Inicializa a figura

    sns.boxplot(x=df[column_name], ax=axes[0]) # Plota o boxplot
    axes[0].set_title(f"Box plot de {column_name}") # Adiciona o título

    axes[1].hist(df[column_name],
                   bins=30,
                   edgecolor="Black",
                   alpha=0.7) # Plota o histograma
    axes[1].set_title(f"Histograma de {column_name}") # Adiciona o título
    axes[1].set_xlabel(column_name) # Adiciona o label do eixo x
    axes[1].set_ylabel("Frequência") # Adiciona o label do eixo y

    plt.tight_layout() # Ajusta o layout
    plt.show() # Exibe o gráfico


    