# AA1-EV03 – Modelamiento de datos

Preprocesamiento del dataset `customer_satisfaction_data.csv` para entrenar un modelo de regresión que predice `SatisfactionScore`.

## Archivos

- `preprocesamiento.py`: código de limpieza, transformación y escalado.
- `customer_satisfaction_data.csv`: dataset crudo (500 registros, 8 variables).
- `customer_satisfaction_modelado.csv`: dataset final listo para el modelo (500 registros, 12 columnas).

## Qué hace el código

1. Unifica `PlanType` en 3 categorías (Basico, Estandar, Premium).
2. Corrige las edades fuera de 18–100 años con la mediana.
3. Imputa los faltantes de `MonthlyBill` con la mediana.
4. Unifica los 5 formatos de `SignupDate` y crea `Antiguedad_Cliente` (días hasta 2024-07-01).
5. Agrupa `LastSupportTicket` en 5 categorías de motivo.
6. Aplica One-Hot Encoding a las variables categóricas (referencias: Basico y Contenido).
7. Estandariza (Z-score) las 4 variables numéricas.

## Cómo ejecutarlo

```bash
pip install pandas numpy
python preprocesamiento.py
```

El script lee `customer_satisfaction_data.csv` de la misma carpeta y genera `customer_satisfaction_modelado.csv`.
