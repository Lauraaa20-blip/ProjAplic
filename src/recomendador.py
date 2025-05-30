# src/recomendador.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. Carregar e preparar os dados
# ---------------------------

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df = df.dropna(subset=['Curso', 'Tipo de Ensino', 'Localidade', 'Categoria'])
    df['Categoria'] = df['Categoria'].fillna('')
    return df

# ---------------------------
# 2. Criar matriz de similaridade
# ---------------------------

def gerar_matriz_similaridade(df):
    tfidf = TfidfVectorizer(stop_words='portuguese')
    matriz = tfidf.fit_transform(df['Categoria'])
    similaridade = cosine_similarity(matriz)
    return similaridade

# ---------------------------
# 3. Função de recomendação
# ---------------------------

def recomendar_cursos(nome_curso, df, similaridade, n=5):
    if nome_curso not in df['Curso'].values:
        print("Curso não encontrado.")
        return []

    idx = df[df['Curso'] == nome_curso].index[0]
    scores = list(enumerate(similaridade[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    indices_recomendados = [i[0] for i in scores[1:n+1]]

    recomendados = df.iloc[indices_recomendados][['Curso', 'Instituição', 'Localidade', 'Tipo de Ensino']]
    return recomendados
