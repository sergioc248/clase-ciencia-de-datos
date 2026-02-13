import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

# 1. Cargar CSV y hacer scatterplot

df = pd.read_csv('codificador/pacientes.csv')

plt.figure()
plt.scatter(df['edad'], df['colesterol'], c=df['problema_cardiaco'])
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.title('Pacientes: Edad vs Colesterol')
plt.show()

# 2. Imprimir maximos y minimos

print("Edad min:", df['edad'].min())
print("Edad max:", df['edad'].max())
print("Colesterol min:", df['colesterol'].min())
print("Colesterol max:", df['colesterol'].max())

# 3. Inputs del usuario

edad = int(input("Digite la edad: "))
colesterol = int(input("Digite el nivel de colesterol: "))

# 4. Cargar scaler y escalar

scaler = joblib.load('codificador/scaler.jb')

valores = scaler.transform([[edad, colesterol]])

edad_scaled = valores[0][0]
colesterol_scaled = valores[0][1]

# 5. Multiplicar valores x2

edad_scaled *= 2
colesterol_scaled *= 2

# 6. Asignar a X1 y X2

X1 = edad_scaled
X2 = colesterol_scaled

# Red neuronal

def forward(X1, X2):
    a1 = 1 / (1 + math.exp(-(0.64 + (-0.021 * X1) + (-1.1 * X2))))
    a2 = 1 / (1 + math.exp(-(0.19 + (0.051 * X1) + (-0.59 * X2))))
    a3 = 1 / (1 + math.exp(-(-0.0050 + (1.2 * X1) + (-0.58 * X2))))
    a4 = 1 / (1 + math.exp(-(-0.36 + (-0.0089 * X1) + (0.83 * X2))))
    a5 = 1 / (1 + math.exp(-(-0.076 + (1.1 * a1) + (0.56 * a2) + (-0.98 * a3) + (-0.85 * a4))))
    a6 = 1 / (1 + math.exp(-(0.072 + (1.0 * a1) + (0.45 * a2) + (-1.1 * a3) + (-1.1 * a4))))
    a7 = 1 / (1 + math.exp(-(0.14 + (1.4 * a1) + (0.83 * a2) + (-1.0 * a3) + (-1.3 * a4))))
    a8 = 1 / (1 + math.exp(-(0.22 + (-0.055 * a1) + (0.026 * a2) + (0.63 * a3) + (0.17 * a4))))
    a9 = 1 / (1 + math.exp(-(0.76 + (-0.94 * a5) + (-1.0 * a6) + (-1.1 * a7) + (0.15 * a8))))
    a10 = 1 / (1 + math.exp(-(-1.3 + (1.3 * a5) + (1.4 * a6) + (2.0 * a7) + (-1.1 * a8))))
    a11 = math.tanh(1.2 + (2.3 * a9) + (-3.3 * a10))
    return a11

# 7. Resultado de la red

resultado = forward(X1, X2)
print("Resultado red neuronal:", resultado)

# 8. Clasificacion

if resultado <= 0:
    pred = 0
else:
    pred = 1

# 9. Mensaje final

if pred == 0:
    print("No sufrirá de problemas cardiacos")
else:
    print("Lo lamento, sufre problemas cardiacos")
