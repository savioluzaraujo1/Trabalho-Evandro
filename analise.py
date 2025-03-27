import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregamento dos dados do arquivo CSV para um DataFrame
df = pd.read_csv("StudentsPerformance.csv")

# Exibe uma amostra inicial dos dados
print("Visualização inicial dos dados:\n", df.head())

# Lista o nome de todas as colunas do conjunto de dados
print("\nColunas disponíveis no dataset:\n", df.columns)

# Verifica se existem dados ausentes em cada coluna
print("\nChecagem de valores ausentes:\n", df.isnull().sum())

# Mostra estatísticas básicas sobre os dados numéricos
print("\nResumo estatístico das variáveis numéricas:\n", df.describe())

# Gráficos para entender como as notas estão distribuídas
plt.figure(figsize=(8, 6))
sns.histplot(df["math score"], bins=20, kde=True, color="blue")
plt.title("Distribuição das Notas em Matemática")
plt.xlabel("Pontuação")
plt.ylabel("Número de Estudantes")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df["reading score"], bins=20, kde=True, color="green")
plt.title("Distribuição das Notas em Leitura")
plt.xlabel("Pontuação")
plt.ylabel("Número de Estudantes")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df["writing score"], bins=20, kde=True, color="red")
plt.title("Distribuição das Notas em Escrita")
plt.xlabel("Pontuação")
plt.ylabel("Número de Estudantes")
plt.show()

# Visualiza como as notas de leitura se relacionam com as de matemática
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["reading score"], y=df["math score"], color="purple")
plt.xlabel("Nota em Leitura")
plt.ylabel("Nota em Matemática")
plt.title("Correlação entre Leitura e Matemática")
plt.show()

# Gera matriz de correlação entre variáveis numéricas
correlacoes = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlacoes, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1)
plt.title("Correlação entre Variáveis Numéricas")
plt.show()

# Cálculo das médias das pontuações em cada disciplina
print("\nMédias gerais das disciplinas:")
print("Matemática:", df["math score"].mean())
print("Leitura:", df["reading score"].mean())
print("Escrita:", df["writing score"].mean())

# Identifica qual variável tem mais influência sobre a nota de matemática
maior_correlacao = correlacoes["math score"].drop("math score").idxmax()
print("\nA variável com maior correlação com a nota de Matemática é:", maior_correlacao)

# Etapa de pré-processamento para aplicação de modelo de ML
# Separa a variável alvo e os atributos explicativos
y = df["math score"]
X = df.drop(columns=["math score"])

# Conversão de colunas categóricas em numéricas com LabelEncoder
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Padroniza os dados numéricos para facilitar o aprendizado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide os dados em subconjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicializa e treina o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Realiza previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliação da performance do modelo usando métricas de erro
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")


# Parte 2 - Análise de Resultados

# Gráfico que compara as notas reais com as previstas pelo modelo
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Notas Originais de Matemática")
plt.ylabel("Notas Estimadas")
plt.title("Comparativo entre Notas Reais e Previstos")
plt.show()

# Avalia quais variáveis mais influenciam o modelo
importances = abs(model.coef_)
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("Impacto relativo de cada variável na previsão da nota de Matemática:")
print(feature_importance)
