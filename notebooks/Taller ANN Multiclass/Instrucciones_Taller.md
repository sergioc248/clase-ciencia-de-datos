Taller de ANN Multiclass
Objetivo del modelo:
Clasificar a los clientes en tres categorías de "Credit Score" (0, 1 y 2), basado en su perfil de crédito. Este es un problema de clasificación multiclase.
Variables:
Variable objetivo (target): Credit Score (0, 1, 2), que indica el riesgo del crédito.
Variables predictoras (features): Son las columnas que describen al cliente, tales como "Age", "Annual_Income", "Num_Bank_Accounts", "Credit_Mix", "Outstanding_Debt", "Monthly_Balance", entre otras.
Preprocesamiento de datos:
Eliminación de columnas irrelevantes: Como "SSN" y "Customer_ID", que no son relevantes para el modelo.
Conversión de variables categóricas: Las columnas como "Credit_Mix" pueden necesitar ser convertidas en valores numéricos o usar codificación one-hot.
Normalización: Es recomendable normalizar las características numéricas para que el modelo pueda entrenar de manera más eficiente.
Reducción de Dimensionalidad con compenentes principales
La lista de columnas que forman parte de un conjunto de datos sobre el perfil financiero de los clientes.
Estas columnas representan varias características relacionadas con los clientes y su comportamiento crediticio. A continuación, te ofrezco una breve explicación de cada columna:
Customer_ID: Identificador único del cliente. Name: Nombre del cliente.
Age: Edad del cliente.
SSN: Número de Seguro Social del cliente (se utiliza en algunos países para identificar a los individuos en registros financieros, aunque debe manejarse con cuidado por razones de privacidad).
Occupation: Ocupación del cliente (por ejemplo, ingeniero, médico, etc.).
Annual_Income: Ingreso anual del cliente.
Monthly_Inhand_Salary: Salario mensual neto (lo que recibe el cliente después de deducciones, como impuestos).
Num_Bank_Accounts: Número de cuentas bancarias que posee el cliente.
Num_Credit_Card: Número de tarjetas de crédito que tiene el cliente.
Interest_Rate: Tasa de interés aplicada en los préstamos del cliente (puede aplicarse a préstamos personales, préstamos hipotecarios, etc.).
Num_of_Loan: Número de préstamos que el cliente ha tomado.
Type_of_Loan: Tipo de préstamo que el cliente tiene (por ejemplo, personal, hipotecario, estudiantil).
Delay_from_due_date: Retraso desde la fecha de vencimiento del pago (por ejemplo, cuántos días de retraso tiene el cliente en el pago).
Num_of_Delayed_Payment: Número de pagos atrasados del cliente en el pasado.
Changed_Credit_Limit: Indica si el límite de crédito del cliente ha sido modificado.
Num_Credit_Inquiries: Número de consultas de crédito realizadas por el cliente.
Credit_Mix: El tipo de mezcla de crédito que el cliente tiene (puede ser una combinación de tarjetas de crédito, préstamos, etc.).
Outstanding_Debt: Deuda pendiente que el cliente tiene actualmente.
Credit_Utilization_Ratio: Relación de utilización de crédito, que es el porcentaje de crédito disponible utilizado por el cliente.
Credit_History_Age: Antigüedad del historial crediticio del cliente.
Payment_of_Min_Amount: Si el cliente realiza al menos el pago mínimo del saldo de su tarjeta de crédito.
Total_EMI_per_month: El pago total mensual de EMI (Equated Monthly Installment), que podría incluir préstamos o financiamientos de productos.
Amount_invested_monthly: Monto invertido mensualmente por el cliente.
Payment_Behaviour: Comportamiento de pago del cliente (por ejemplo, si paga a tiempo, si tiene pagos atrasados, etc.).
Monthly_Balance: Balance mensual en las cuentas bancarias del cliente.
Credit_Score: El puntaje de crédito del cliente, que evalúa su comportamiento crediticio y riesgo de crédito. Es la columna etiquetada.
 

Data: riesgo.csv
 