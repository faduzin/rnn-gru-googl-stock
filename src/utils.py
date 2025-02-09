import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print('Data loaded successfully.')
        return data
    except FileNotFoundError:
        print('File not found.')
        return None
    

def save_data(df, file_path):
    try: # Tenta salvar o arquivo
        df.to_csv(file_path, index=False) # Salva o dataframe
        print(f"Arquivo salvo em: {file_path}.") # Exibe mensagem de sucesso
    except: # Se houver erro
        raise("Falha ao salvar arquivo.") # Exibe mensagem de erro


def data_info(data):
    print(data.info())
    print("-" * 50)
    print(data.describe())
    print("-" * 50)
    print(data.head(5))
    print("-" * 50)
    print("Data shape: ",data.shape)
    print("Amount of duplicates: ", data.duplicated().sum())


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_correlations(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

