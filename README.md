# Tarea 3: Introducción a Series de Tiempo

**Curso:** Análisis y Ciencia de Datos  
**Estudiante:** Daniela Natareno - dnatareno@unis.edu.gt

## Descripción del proyecto

Este repositorio contiene el desarrollo de la Tarea 3 del curso de Análisis y Ciencia de Datos, enfocada en el análisis de series de tiempo y su aplicación a datos de ventas minoristas.

Se utiliza el conjunto de datos **Online Retail**, que contiene transacciones comerciales realizadas entre diciembre de 2010 y diciembre de 2011.

La tarea incluye demostraciones y simulaciones estadísticas, limpieza y exploración de datos, construcción de una serie temporal de ventas netas diarias, análisis de tendencia y estacionalidad, ajuste de modelos estadísticos, evaluación de pronósticos y segmentación de clientes mediante RFM.

Los procedimientos fueron implementados en Python y los resultados se documentaron en un informe elaborado en LaTeX.

## Estructura de tarea

### Parte 0: Fundamentos teóricos y simulaciones

Incluye demostraciones matemáticas y simulaciones para estudiar las propiedades de procesos autorregresivos AR(1), el comportamiento de la media ante cambios de tendencia y la prueba de Ljung-Box.

### Parte 1: Exploración y limpieza de datos

Se realiza la exploración inicial del conjunto de datos Online Retail y la identificación de anomalías, registros duplicados, cancelaciones y movimientos administrativos.

Se implementa una función reutilizable de limpieza y se genera una bitácora que documenta las reglas aplicadas.

**Script principal:** `Parte 1/limpieza.py`

### Parte 2: Construcción de la serie temporal

Se construye una serie de ventas netas diarias a partir de las transacciones limpias.

Se identifican días sin transacciones, se investigan posibles cierres y feriados del Reino Unido y se revisa si el último día del conjunto de datos contiene información incompleta.

**Script principal:** `Parte 2/calendario.py`

### Parte 3: Tendencia, estacionalidad y autocorrelación

Se analiza el comportamiento temporal de las ventas mediante una media móvil centrada de siete días, índices estacionales por día de la semana y la función de autocorrelación.

También se realiza un diagnóstico exploratorio de dependencia temporal utilizando el estadístico de Ljung-Box.

**Script principal:** `Parte 3/tendencia.py`

### Parte 4: Modelos estadísticos

Se implementan diferentes modelos para analizar la estructura temporal de las ventas:

- Regresión OLS por día de la semana.
- Modelo AR(1) sobre los residuos de OLS.
- Modelos SARIMA S1 y S2.
- Diagnósticos de autocorrelación de residuos.
- Pronósticos a largo plazo.

**Script principal:** `Parte 4/modelos.py`

### Parte 5: Pronósticos y evaluación

Se implementan seis modelos de pronóstico:

- **B1:** promedio histórico de ventas.
- **B2:** última venta observada del mismo día de la semana.
- **M1:** regresión con tendencia y día de la semana.
- **M2:** regresión con indicadores semanales y rezagos históricos.
- **S1:** SARIMA(1,0,0)(1,0,0,6).
- **S2:** SARIMA(0,1,1)(0,1,1,6).

Los modelos se evalúan mediante MAE y RMSE, utilizando validación cruzada aleatoria y validación temporal con origen móvil.

También se desarrolla un experimento para demostrar cómo la fuga de información puede afectar la evaluación de los modelos.

**Script principal:** `Parte 5/pronosticos.py`

### Parte 6: Análisis RFM de clientes

Se estudia el comportamiento de los clientes mediante las variables:

- **Recencia (R):** tiempo transcurrido desde la última compra.
- **Frecuencia (F):** número de compras realizadas.
- **Monto (M):** valor monetario neto de las transacciones.

Se asignan quintiles RFM, se evalúa la recompra mediante AUC, se identifican clientes Champions y se calcula la participación de clientes nuevos en las ventas.

**Scripts:** `Parte 6/clientes.py` y `Parte 6/graficas.py`


## Requisitos e instalación

El proyecto fue desarrollado en Python y utiliza las librerías especificadas en el archivo `requirements.txt`.

Para reproducir los análisis, se recomienda crear un entorno virtual e instalar las dependencias.

**Crear un entorno virtual:**

```bash
python -m venv .venv
```

**Activar el entorno virtual en Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

**Instalar las dependencias:**

```bash
python -m pip install -r requirements.txt
```

Una vez instaladas las dependencias, se pueden ejecutar los scripts en el orden indicado en la sección de ejecución del proyecto.

## Ejecución del proyecto

Los scripts deben ejecutarse desde la carpeta principal del repositorio, siguiendo el orden de las partes:

```bash
python "Parte 1/limpieza.py"
python "Parte 2/calendario.py"
python "Parte 3/tendencia.py"
python "Parte 4/modelos.py"
python "Parte 5/pronosticos.py"
python "Parte 6/clientes.py"
python "Parte 6/graficas.py"
```

La Parte 0 contiene scripts independientes para las simulaciones estadísticas y puede ejecutarse por separado.

Es importante ejecutar primero las Partes 1 y 2, ya que las partes posteriores utilizan los archivos generados durante la preparación de datos.

## Consideraciones metodológicas

Para el análisis de ventas se conservan las cancelaciones comerciales válidas, de modo que los importes reflejen ventas netas.

Los días sin transacciones se mantienen como valores faltantes cuando no existe evidencia suficiente para tratarlos como ventas iguales a cero.

Los modelos de pronóstico se evalúan respetando una separación temporal entre entrenamiento y prueba. El período de entrenamiento termina el 30 de septiembre de 2011 y la evaluación posterior se realiza hasta el 8 de diciembre de 2011.

En los modelos SARIMA se utiliza una secuencia de días con ventas observadas, mientras que otros análisis conservan explícitamente los rezagos en días calendario.

## Resultados

Cada parte del la tarea cuenta con una carpeta `Resultados`, donde se almacenan los archivos CSV y las gráficas generadas durante la ejecución.

Entre los principales resultados se encuentran las series de ventas diarias, índices estacionales, funciones de autocorrelación, parámetros de modelos estadísticos, comparaciones de errores de pronóstico y métricas de comportamiento de clientes.

El informe final presenta el procedimiento, los resultados y su interpretación.

## Fuente de datos

**UCI Machine Learning Repository — Online Retail**

https://archive.ics.uci.edu/dataset/352/online+retail

## Documentación

El repositorio incluye el informe final en PDF, su código fuente en LaTeX y las gráficas utilizadas, donde se desarrollan los ejercicios, las demostraciones, los análisis estadísticos y las conclusiones de la tarea.