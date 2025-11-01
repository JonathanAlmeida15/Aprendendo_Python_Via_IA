"""
projeto_integrador.py
Projeto Integrador - Previsão de Valor de Casas (California Housing)
Gera:
 - pastas: outputs/, outputs/figs/, outputs/models/
 - arquivos: metrics.csv, feature_importances.csv, report.md
 - modelos salvos (.joblib)
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------------------------
# Config / diretórios
# ---------------------------
OUTDIR = "outputs"
FIGDIR = os.path.join(OUTDIR, "figs")
MODELDIR = os.path.join(OUTDIR, "models")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# ---------------------------
# 1. Coleta (carrega do scikit-learn)
# ---------------------------
print("1. Coletando dados...")
dataset = fetch_california_housing(as_frame=True)
df = dataset.frame.copy()
# alvo
TARGET = "MedHouseVal"  # scikit-learn usa MedHouseVal (capitalization normal)
print("Dados carregados. Shape:", df.shape)

# ---------------------------
# 2. Limpeza e inspeção
# ---------------------------
print("2. Inspeção rápida")
print(df.head())
print("\nInfos:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe().T)

# Verificar nulos
nulos = df.isnull().sum()
print("\nNulos por coluna:\n", nulos)

# Observação: esse dataset não possui nulos por padrão; se houvesse, faríamos:
# df = df.fillna(df.median())

# ---------------------------
# 3. Feature engineering simples
# ---------------------------
print("3. Feature engineering")
# criar features exemplo: rooms_per_household, bedrooms_per_room, population_per_household
df["rooms_per_household"] = df["AveRooms"] / (df["AveOccup"] + 1e-6)
df["bedrooms_per_room"] = df["AveBedrms"] / (df["AveRooms"] + 1e-6)
df["population_per_household"] = df["Population"] / (df["AveOccup"] + 1e-6)

# salvar snapshot dos dados processados
processed_csv = os.path.join(OUTDIR, "california_processed.csv")
df.to_csv(processed_csv, index=False)
print("Dados processados salvos em:", processed_csv)

# ---------------------------
# 4. EDA (gráficos salvos)
# ---------------------------
print("4. Gerando EDA e gráficos...")

# histograma do target
plt.figure(figsize=(8,4))
sns.histplot(df[TARGET], bins=40, kde=True)
plt.title("Distribuição do Target (MedHouseVal)")
plt.xlabel("MedHouseVal (100k USD)")
plt.tight_layout()
hist_target = os.path.join(FIGDIR, "hist_target.png")
plt.savefig(hist_target)
plt.close()

# correlação (heatmap)
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de Correlação")
plt.tight_layout()
heatmap = os.path.join(FIGDIR, "correlation_matrix.png")
plt.savefig(heatmap)
plt.close()

# scatter: MedInc x MedHouseVal
plt.figure(figsize=(7,5))
sns.scatterplot(x="MedInc", y=TARGET, data=df, alpha=0.4)
plt.title("MedInc vs MedHouseVal")
plt.tight_layout()
scatter1 = os.path.join(FIGDIR, "medinc_vs_target.png")
plt.savefig(scatter1)
plt.close()

# boxplot para outliers (exemplo: AveRooms)
plt.figure(figsize=(6,4))
sns.boxplot(x=df["AveRooms"])
plt.title("Boxplot - AveRooms")
plt.tight_layout()
boxplot1 = os.path.join(FIGDIR, "boxplot_averooms.png")
plt.savefig(boxplot1)
plt.close()

print("Gráficos salvos em:", FIGDIR)

# ---------------------------
# 5. Preparar X, y e split
# ---------------------------
print("5. Preparando dados para modelagem...")
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# padronização (salvaremos o scaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# salvar scaler
scaler_path = os.path.join(MODELDIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)

# ---------------------------
# 6. Treinar modelos
# ---------------------------
print("6. Treinando modelos...")

models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []

for name, model in models.items():
    print(f" Treinando: {name} ...")
    model.fit(X_train_scaled, y_train)
    # previsões
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # cross-val (5 folds) em dados de treino
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2", n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    # salvar modelo
    model_path = os.path.join(MODELDIR, f"{name}.joblib")
    joblib.dump(model, model_path)
    # feature importances (se disponível)
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        feat_imp = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": X.columns,
            "importance": feat_imp
        }).sort_values("importance", ascending=False)
        fi_csv = os.path.join(OUTDIR, f"feature_importances_{name}.csv")
        fi_df.to_csv(fi_csv, index=False)
    else:
        fi_df = None

    results.append({
        "model": name,
        "rmse": float(rmse),
        "mse": float(mse),
        "r2": float(r2),
        "cv_mean_r2": float(cv_mean),
        "cv_std_r2": float(cv_std),
        "model_path": model_path,
        "feature_importances_csv": fi_csv if feat_imp is not None else None
    })
    print(f"  -> {name} done. RMSE: {rmse:.4f}, R2: {r2:.4f}, CV_R2: {cv_mean:.4f}±{cv_std:.4f}")

# salvar métricas em CSV
metrics_df = pd.DataFrame(results)
metrics_csv = os.path.join(OUTDIR, "metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
print("Métricas salvas em:", metrics_csv)

# ---------------------------
# 7. Comparações visuais: Pred vs Real (para Random Forest)
# ---------------------------
rf = joblib.load(os.path.join(MODELDIR, "random_forest.joblib"))
y_pred_rf = rf.predict(X_test_scaled)

plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred_rf, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("Real vs Previsto (Random Forest)")
plt.tight_layout()
predscatter = os.path.join(FIGDIR, "real_vs_pred_rf.png")
plt.savefig(predscatter)
plt.close()

# ---------------------------
# 8. Exportar relatório simples (Markdown + imagens)
# ---------------------------
print("8. Gerando relatório (report.md)...")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_md = os.path.join(OUTDIR, "report.md")
with open(report_md, "w", encoding="utf-8") as f:
    f.write(f"# Relatório - Projeto Integrador (Previsão de Casas)\n")
    f.write(f"_Gerado em: {now}_\n\n")
    f.write("## 1. Sumário dos Dados\n")
    f.write(f"- Registros: {df.shape[0]}\n")
    f.write(f"- Features: {df.shape[1]}\n\n")
    f.write("## 2. Métricas dos Modelos\n\n")
    f.write(metrics_df.to_markdown(index=False))
    f.write("\n\n## 3. Gráficos\n\n")
    for img in [hist_target, heatmap, scatter1, boxplot1, predscatter]:
        f.write(f"![{os.path.basename(img)}]({os.path.relpath(img, OUTDIR)})\n\n")

print("Relatório gerado em:", report_md)

# ---------------------------
# 9. Mensagem final
# ---------------------------
print("\nTudo concluído. Artefatos salvos em:", OUTDIR)
print(" - gráficos:", FIGDIR)
print(" - modelos:", MODELDIR)
print(" - report:", report_md)
