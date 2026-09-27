"""Preprocesamiento de customer_satisfaction_data.csv (Evidencia AA1-EV03)."""
import re
import numpy as np
import pandas as pd
from datetime import datetime

df = pd.read_csv("customer_satisfaction_data.csv", encoding="utf-8-sig")

# 1. PlanType: unificar categorias
mapa_plan = {"basico": "Basico", "estandar": "Estandar", "premium": "Premium", "prem": "Premium"}
def norm_plan(x):
    x = x.strip().lower().translate(str.maketrans("áéíóú", "aeiou"))
    return mapa_plan[x]
df["PlanType"] = df["PlanType"].map(norm_plan)

# 2. Age: fuera de rango de negocio (<18 o >100) -> NaN -> mediana
df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan
df["Age"] = df["Age"].fillna(df["Age"].median())

# 3. MonthlyBill: imputar con la mediana
df["MonthlyBill"] = df["MonthlyBill"].fillna(df["MonthlyBill"].median())

# 4. SignupDate: parseo por patron explicito (5 formatos)
def parse_fecha(s):
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):  return datetime.strptime(s, "%Y-%m-%d")
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", s):  return datetime.strptime(s, "%Y/%m/%d")
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):  return datetime.strptime(s, "%d/%m/%Y")
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s):  return datetime.strptime(s, "%m-%d-%Y")
    if re.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{4}", s): return datetime.strptime(s, "%d-%b-%Y")
    raise ValueError(s)
df["SignupDate"] = pd.to_datetime(df["SignupDate"].map(parse_fecha))
FECHA_REF = df["SignupDate"].max() + pd.Timedelta(days=1)
df["Antiguedad_Cliente"] = (FECHA_REF - df["SignupDate"]).dt.days

# 5. LastSupportTicket: categoria de motivo por palabras clave
reglas = [
    ("Positivo",    ["me encanta", "excelente", "muy buen contenido"]),
    ("Tecnico",     ["calidad de video", "congela", "error 503", "se cierra", "acceder"]),
    ("Facturacion", ["cobro", "pago", "precio"]),
    ("Contenido",   ["no encuentro", "subtítulo"]),
    ("Servicio",    ["servicio al cliente", "cambiar mi plan"]),
]
def cat_ticket(t):
    t = t.lower()
    for cat, kws in reglas:
        if any(k in t for k in kws): return cat
    return "Otro"
df["Ticket_Categoria"] = df["LastSupportTicket"].map(cat_ticket)

# 6. One-Hot Encoding (drop_first evita la trampa de variables ficticias)
#    Referencias: Basico (plan) y Contenido (ticket)
X = pd.get_dummies(df[["PlanType", "Ticket_Categoria"]], drop_first=True, dtype=int)
X.columns = [c.replace("Ticket_Categoria_", "Ticket_") for c in X.columns]

# 7. Estandarizacion Z-score
num = ["Age", "MonthlyBill", "HoursWatched_Last30d", "Antiguedad_Cliente"]
nombres = {"Age": "Age_scaled", "MonthlyBill": "MonthlyBill_scaled",
           "HoursWatched_Last30d": "HoursWatched_scaled",
           "Antiguedad_Cliente": "Antiguedad_Cliente_scaled"}
stats = {}
for c in num:
    mu, sd = df[c].mean(), df[c].std(ddof=0)
    stats[c] = (mu, sd)
    X[nombres[c]] = ((df[c] - mu) / sd).round(4)

orden = list(nombres.values()) + ["PlanType_Estandar", "PlanType_Premium",
         "Ticket_Tecnico", "Ticket_Facturacion", "Ticket_Servicio", "Ticket_Positivo"]
final = X[orden].copy()
final.insert(0, "CustomerID", df["CustomerID"])
final["SatisfactionScore"] = df["SatisfactionScore"]
final.to_csv("customer_satisfaction_modelado.csv", index=False)

if __name__ == "__main__":
    print(df["PlanType"].value_counts()); print(df["Ticket_Categoria"].value_counts())
    print("Fecha ref", FECHA_REF.date(), df.SignupDate.min().date(), df.SignupDate.max().date())
    print(df[num].describe().round(2).to_string())
    for c,(mu,sd) in stats.items(): print(c, round(mu,4), round(sd,4))
    print(list(final.columns), final.shape, final.isna().sum().sum())
    print(final.head(3).to_string())
    print("mono dates:", df.SignupDate.is_monotonic_increasing)
