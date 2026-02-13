import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import json
 
 
# 1. Cargar CSV
df = pd.read_csv("codificador/pacientes.csv", sep=",")
 
 
# 2. Imputar valores faltantes
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
 
 
# 2.1 Modificar problema_cardiaco: 0 -> -1
df_imputed["problema_cardiaco"] = df_imputed["problema_cardiaco"].replace(0, -1)
 
 
# 3. Separar features y label
X = df_imputed.drop(columns=["problema_cardiaco"])
y = df_imputed["problema_cardiaco"]
 
 
# 4. Escalar SOLO las features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
 
# Guardar el scaler entrenado
joblib.dump(scaler, "codificador/scaler.jb")
 
 
# 4.1 Multiplicar edad y colesterol por 2
X_scaled["edad"] *= 2
X_scaled["colesterol"] *= 2
 
 
# 5. Reconstruir DataFrame final
df_scaled = X_scaled.copy()
df_scaled["problema_cardiaco"] = y
 
 
# 6. Convertir al formato solicitado
output = [
  {
      "x": row["edad"],
      "y": row["colesterol"],
      "label": int(row["problema_cardiaco"])
      }
  for _, row in df_scaled.iterrows()
]
 
 
# 7. Exportar a JSON
output_file = "codificador/pacientes_transformados.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)
 
 
print(f"Archivo exportado en: {output_file}")