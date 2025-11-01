# Relatório - Projeto Integrador (Previsão de Casas)
_Gerado em: 2025-11-01 02:26:18_

## 1. Sumário dos Dados
- Registros: 20640
- Features: 12

## 2. Métricas dos Modelos

| model             |     rmse |      mse |       r2 |   cv_mean_r2 |   cv_std_r2 | model_path                              | feature_importances_csv                       |
|:------------------|---------:|---------:|---------:|-------------:|------------:|:----------------------------------------|:----------------------------------------------|
| linear_regression | 0.673817 | 0.45403  | 0.653521 |     0.659342 |  0.0195038  | outputs\models\linear_regression.joblib |                                               |
| random_forest     | 0.505037 | 0.255063 | 0.805356 |     0.803367 |  0.00571213 | outputs\models\random_forest.joblib     | outputs\feature_importances_random_forest.csv |

## 3. Gráficos

![hist_target.png](figs\hist_target.png)

![correlation_matrix.png](figs\correlation_matrix.png)

![medinc_vs_target.png](figs\medinc_vs_target.png)

![boxplot_averooms.png](figs\boxplot_averooms.png)

![real_vs_pred_rf.png](figs\real_vs_pred_rf.png)

