import pandas as pd
import numpy as np

def calcular_regressao(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    
    area = df['area_util_m2'].to_list()
    quartos = df['quantidade_quartos'].to_list()
    preco = df['preco_venda'].to_list()
    
    n = len(preco)
    X = np.column_stack((np.ones(n), area, quartos))
    y = np.array(preco)
    
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta.tolist()
