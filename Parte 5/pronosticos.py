import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO_SERIE = CARPETA_PROYECTO / "Parte 2" / "Resultados" / "serie_diaria_sin_ultimo_dia.csv"
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"

FECHA_FIN_ENTRENAMIENTO = "2011-09-30"
FECHA_FIN_PRUEBA = "2011-12-08"

DIAS_SEMANA = [0, 1, 2, 3, 4, 6]

def cargar_serie():
    if not ARCHIVO_SERIE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO_SERIE}")

    serie = pd.read_csv(
        ARCHIVO_SERIE,
        parse_dates=["Fecha"],
        index_col="Fecha"
    )

    return serie.sort_index()

def preparar_datos():
    serie = cargar_serie()

    entrenamiento = serie.loc[:FECHA_FIN_ENTRENAMIENTO].copy()
    prueba = serie.loc["2011-10-03":FECHA_FIN_PRUEBA].copy()

    entrenamiento = entrenamiento.dropna(subset=["Venta_neta"])
    prueba = prueba.dropna(subset=["Venta_neta"])

    return serie, entrenamiento, prueba

def pronosticar_b1(historial, fechas_objetivo):
    # B1 utiliza la media de todas las ventas observadas.
    ventas = historial["Venta_neta"].dropna()
    promedio = ventas.mean()

    return pd.Series(
        promedio,
        index=fechas_objetivo,
        name="B1"
    )

def pronosticar_b2(historial, fechas_objetivo):
    # B2 busca la venta más reciente del mismo día de la semana.
    ventas = historial["Venta_neta"].dropna()
    pronosticos = {}

    for fecha in fechas_objetivo:
        mismo_dia = ventas.loc[
            ventas.index.dayofweek == fecha.dayofweek
        ]

        if mismo_dia.empty:
            raise ValueError(
                f"No existe historial para pronosticar {fecha.date()}"
            )

        pronosticos[fecha] = mismo_dia.iloc[-1]

    return pd.Series(pronosticos, name="B2")

def construir_variables_m1(fechas, fecha_inicial):
    # La tendencia representa los días calendario desde el inicio.
    tendencia = (
        pd.DatetimeIndex(fechas) - fecha_inicial
    ).days.astype(float)

    variables = pd.DataFrame(
        {"Tendencia": tendencia},
        index=fechas
    )

    # Se utilizan cinco indicadoras y el lunes como referencia.
    for dia in DIAS_SEMANA[1:]:
        variables[f"Dia_{dia}"] = (
            pd.DatetimeIndex(fechas).dayofweek == dia
        ).astype(float)

    return sm.add_constant(variables, has_constant="add")

def pronosticar_m1(historial, fechas_objetivo):
    # M1 combina una tendencia lineal con el día de la semana.
    ventas = historial["Venta_neta"].dropna()
    fecha_inicial = ventas.index.min()

    X = construir_variables_m1(ventas.index, fecha_inicial)
    y = ventas.astype(float)

    modelo = sm.OLS(y, X).fit()

    X_futuro = construir_variables_m1(
        fechas_objetivo,
        fecha_inicial
    )

    predicciones = modelo.predict(X_futuro)

    return pd.Series(
        np.asarray(predicciones),
        index=fechas_objetivo,
        name="M1"
    )

def buscar_venta_anterior(ventas, fecha, semanas):
    # Busca el mismo día de la semana en semanas anteriores.
    fecha_busqueda = fecha - pd.Timedelta(days=7 * semanas)

    while fecha_busqueda >= ventas.index.min():
        if fecha_busqueda in ventas.index:
            return float(ventas.loc[fecha_busqueda])

        fecha_busqueda -= pd.Timedelta(days=7)

    return np.nan

def construir_variables_m2(ventas, fechas):
    # M2 utiliza indicadores semanales y dos rezagos históricos.
    variables = pd.DataFrame(index=fechas)

    for dia in DIAS_SEMANA:
        variables[f"Dia_{dia}"] = (
            pd.DatetimeIndex(fechas).dayofweek == dia
        ).astype(float)

    variables["Rezago_7"] = [
        buscar_venta_anterior(ventas, fecha, 1)
        for fecha in fechas
    ]

    variables["Rezago_14"] = [
        buscar_venta_anterior(ventas, fecha, 2)
        for fecha in fechas
    ]

    return variables

def pronosticar_m2(historial, fechas_objetivo):
    # M2 se ajusta únicamente con información histórica.
    ventas = historial["Venta_neta"].dropna().astype(float)

    X = construir_variables_m2(ventas, ventas.index)
    datos = X.copy()
    datos["Venta_neta"] = ventas

    # Las primeras filas sin rezagos completos no se utilizan.
    datos = datos.dropna()

    X_entrenamiento = datos.drop(columns=["Venta_neta"])
    y_entrenamiento = datos["Venta_neta"]

    modelo = sm.OLS(
        y_entrenamiento,
        X_entrenamiento
    ).fit()

    # Los rezagos del pronóstico solo pueden consultar el historial.
    X_futuro = construir_variables_m2(
        ventas,
        fechas_objetivo
    )

    if X_futuro.isna().any().any():
        raise ValueError("No hay suficientes rezagos para M2.")

    predicciones = modelo.predict(X_futuro)

    return pd.Series(
        np.asarray(predicciones),
        index=fechas_objetivo,
        name="M2"
    )

def ajustar_sarima(historial, nombre):
    # Utiliza únicamente las ventas observadas hasta el origen.
    ventas = historial["Venta_neta"].dropna().astype(float)

    if (ventas <= 0).any():
        raise ValueError("SARIMA requiere ventas positivas para aplicar logaritmos.")

    valores = np.log(ventas.to_numpy())

    # Mantiene las configuraciones utilizadas en la Parte 4.
    if nombre == "S1":
        orden = (1, 0, 0)
        orden_estacional = (1, 0, 0, 6)
        tendencia = "c"
    elif nombre == "S2":
        orden = (0, 1, 1)
        orden_estacional = (0, 1, 1, 6)
        tendencia = "n"
    else:
        raise ValueError(f"Modelo SARIMA desconocido: {nombre}")

    modelo = SARIMAX(
        valores,
        order=orden,
        seasonal_order=orden_estacional,
        trend=tendencia,
        enforce_stationarity=True,
        enforce_invertibility=True
    )

    # Intenta primero el ajuste utilizado en la Parte 4.
    resultado = modelo.fit(
        disp=False,
        maxiter=500,
        method="lbfgs"
    )

    convergio = resultado.mle_retvals.get("converged", False)

    if not convergio:
        print(
            f"Advertencia: {nombre} no convergió con L-BFGS. "
            "Intentando con Powell..."
        )

        # Reintenta sin modificar el orden ni la tendencia SARIMA.
        resultado = modelo.fit(
            disp=False,
            maxiter=1000,
            method="powell"
        )

        convergio = resultado.mle_retvals.get("converged", False)

    if not convergio:
        raise RuntimeError(
            f"El modelo {nombre} no convergió con ninguno de los métodos."
        )

    return resultado

def pronosticar_sarima(historial, fechas_objetivo, nombre):
    # Ajusta el modelo y pronostica los siguientes días abiertos.
    modelo = ajustar_sarima(historial, nombre)

    fechas_objetivo = pd.DatetimeIndex(fechas_objetivo)
    pasos = len(fechas_objetivo)

    pronostico_log = modelo.get_forecast(
        steps=pasos
    ).predicted_mean

    # Regresa los pronósticos a la escala original de ventas.
    pronostico_ventas = np.exp(
        np.asarray(pronostico_log)
    )

    return pd.Series(
        pronostico_ventas,
        index=fechas_objetivo,
        name=nombre
    )

def probar_seis_modelos():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie, entrenamiento, prueba = preparar_datos()

    # Se utiliza el domingo 2 de octubre como primer origen de prueba.
    origen = pd.Timestamp("2011-10-02")
    historial = serie.loc[:origen].copy()

    # Se pronostican únicamente los días abiertos de la siguiente semana.
    fechas_objetivo = prueba.loc[
        "2011-10-03":"2011-10-09"
    ].index

    resultados = pd.DataFrame(index=fechas_objetivo)
    resultados["Venta_real"] = prueba.loc[
        fechas_objetivo, "Venta_neta"
    ]

    # Genera los pronósticos de los cuatro modelos básicos.
    resultados["B1"] = pronosticar_b1(
        historial, fechas_objetivo
    )
    resultados["B2"] = pronosticar_b2(
        historial, fechas_objetivo
    )
    resultados["M1"] = pronosticar_m1(
        historial, fechas_objetivo
    )
    resultados["M2"] = pronosticar_m2(
        historial, fechas_objetivo
    )

    # Ajusta los modelos SARIMA con el historial disponible.
    resultados["S1"] = pronosticar_sarima(
        historial, fechas_objetivo, "S1"
    )
    resultados["S2"] = pronosticar_sarima(
        historial, fechas_objetivo, "S2"
    )

    resultados.index.name = "Fecha"

    print("\nEjercicio 5.1: Seis modelos de pronóstico")
    print(f"Origen: {origen.date()}")
    print(f"Días pronosticados: {len(resultados)}")

    print("\nPronósticos de la primera semana de prueba:")
    print(resultados.round(2).to_string())

    resultados.to_csv(
        CARPETA_RESULTADOS / "prueba_seis_modelos.csv",
        encoding="utf-8-sig"
    )

def calcular_metricas(reales, pronosticos):
    # Calcula los errores en la escala original de ventas.
    mae = mean_absolute_error(reales, pronosticos)
    rmse = np.sqrt(mean_squared_error(reales, pronosticos))

    return mae, rmse

def validar_aleatoriamente():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie, entrenamiento, prueba = preparar_datos()
    datos = entrenamiento.copy().sort_index()

    semillas = range(10)
    modelos = ["B1", "M1", "M2"]
    resultados = []

    print("\nEjercicio 5.2: Validación cruzada aleatoria")
    print("Particiones por semilla: 5")
    print("Cantidad de semillas: 10")

    for semilla in semillas:
        # Divide aleatoriamente los días de entrenamiento.
        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=semilla
        )

        predicciones_semilla = {
            nombre: {"reales": [], "pronosticos": []}
            for nombre in modelos
        }

        for numero_fold, (indices_train, indices_val) in enumerate(
            kfold.split(datos), start=1
        ):
            historial = datos.iloc[indices_train].sort_index()
            validacion = datos.iloc[indices_val].sort_index()

            # Comprueba qué fechas tienen ambos rezagos disponibles.
            ventas_historial = historial["Venta_neta"].dropna()
            variables_m2 = construir_variables_m2(
                ventas_historial,
                validacion.index
            )

            fechas_validas = variables_m2.dropna().index

            if len(fechas_validas) == 0:
                continue

            # Evalúa todos los modelos sobre las mismas fechas.
            reales = validacion.loc[
                fechas_validas, "Venta_neta"
            ].astype(float)

            predicciones = {
                "B1": pronosticar_b1(
                    historial, fechas_validas
                ),
                "M1": pronosticar_m1(
                    historial, fechas_validas
                ),
                "M2": pronosticar_m2(
                    historial, fechas_validas
                )
            }

            for nombre in modelos:
                predicciones_semilla[nombre]["reales"].extend(
                    reales.to_numpy()
                )
                predicciones_semilla[nombre]["pronosticos"].extend(
                    predicciones[nombre].to_numpy()
                )

        # Calcula las métricas con todas las predicciones de la semilla.
        for nombre in modelos:
            reales = np.array(
                predicciones_semilla[nombre]["reales"]
            )

            pronosticos = np.array(
                predicciones_semilla[nombre]["pronosticos"]
            )

            mae, rmse = calcular_metricas(
                reales,
                pronosticos
            )

            resultados.append({
                "Semilla": semilla,
                "Modelo": nombre,
                "Observaciones": len(reales),
                "MAE": mae,
                "RMSE": rmse
            })

        print(f"Semilla {semilla}: completada")

    resultados = pd.DataFrame(resultados)

    resumen = (
        resultados.groupby("Modelo")[["MAE", "RMSE"]]
        .agg(["mean", "std"])
        .round(2)
    )

    print("\nResultados por modelo:")
    print(resumen.to_string())

    print("\nCantidad de observaciones evaluadas:")
    print(
        resultados.groupby("Modelo")["Observaciones"]
        .agg(["min", "max"])
        .to_string()
    )

    resultados.to_csv(
        CARPETA_RESULTADOS / "validacion_aleatoria_detalle.csv",
        index=False,
        encoding="utf-8-sig"
    )

    resumen.to_csv(
        CARPETA_RESULTADOS / "validacion_aleatoria_resumen.csv",
        encoding="utf-8-sig"
    )

    return resultados, resumen

def validar_origen_movil(fecha_inicio, fecha_fin, nombre):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie = cargar_serie()
    inicio = pd.Timestamp(fecha_inicio)
    fin = pd.Timestamp(fecha_fin)

    # El primer origen es el domingo anterior al inicio.
    primer_origen = inicio - pd.Timedelta(
        days=inicio.dayofweek + 1
    )

    origenes = pd.date_range(
        start=primer_origen,
        end=fin,
        freq="W-SUN"
    )

    modelos = ["B1", "B2", "M1", "M2", "S1", "S2"]
    registros = []

    print(f"\nEjercicio 5.2: Validación temporal {nombre}")

    for origen in origenes:
        # El historial incluye las ventas observadas del domingo.
        historial = serie.loc[:origen].copy()

        # La semana siguiente termina el domingo posterior.
        inicio_semana = origen + pd.Timedelta(days=1)
        fin_semana = min(
            origen + pd.Timedelta(days=7),
            fin
        )

        if fin_semana < inicio:
            continue

        fechas_objetivo = serie.loc[
            max(inicio_semana, inicio):fin_semana
        ].dropna(subset=["Venta_neta"]).index

        if len(fechas_objetivo) == 0:
            continue

        reales = serie.loc[
            fechas_objetivo, "Venta_neta"
        ].astype(float)

        # Cada modelo utiliza el historial disponible en el origen.
        predicciones = {
            "B1": pronosticar_b1(historial, fechas_objetivo),
            "B2": pronosticar_b2(historial, fechas_objetivo),
            "M1": pronosticar_m1(historial, fechas_objetivo),
            "M2": pronosticar_m2(historial, fechas_objetivo),
            "S1": pronosticar_sarima(historial, fechas_objetivo, "S1"),
            "S2": pronosticar_sarima(historial, fechas_objetivo, "S2")
        }

        for fecha in fechas_objetivo:
            registro = {
                "Origen": origen,
                "Fecha": fecha,
                "Venta_real": reales.loc[fecha]
            }

            for modelo in modelos:
                registro[modelo] = predicciones[modelo].loc[fecha]

            registros.append(registro)

        print(
            f"Origen {origen.date()}: "
            f"{len(fechas_objetivo)} días pronosticados"
        )

    detalle = pd.DataFrame(registros)

    if detalle.empty:
        raise ValueError("No se generaron pronósticos.")

    resumen = []

    # Calcula las métricas agrupando todos los pronósticos diarios.
    for modelo in modelos:
        mae, rmse = calcular_metricas(
            detalle["Venta_real"],
            detalle[modelo]
        )

        resumen.append({
            "Modelo": modelo,
            "Observaciones": len(detalle),
            "MAE": mae,
            "RMSE": rmse
        })

    resumen = pd.DataFrame(resumen)

    print(f"\nResultados de validación temporal {nombre}:")
    print(resumen.round(2).to_string(index=False))

    detalle.to_csv(
        CARPETA_RESULTADOS / f"validacion_{nombre}_detalle.csv",
        index=False,
        encoding="utf-8-sig"
    )

    resumen.to_csv(
        CARPETA_RESULTADOS / f"validacion_{nombre}_resumen.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return detalle, resumen

def graficar_mejor_modelo():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    archivo = (
        CARPETA_RESULTADOS /
        "validacion_temporal_prueba_detalle.csv"
    )

    datos = pd.read_csv(
        archivo,
        parse_dates=["Fecha"]
    ).sort_values("Fecha")

    # S2 obtuvo el menor MAE y RMSE en prueba.
    plt.figure(figsize=(13, 6))

    plt.plot(
        datos["Fecha"],
        datos["Venta_real"],
        label="Ventas reales",
        linewidth=1.8,
        marker="o",
        markersize=3
    )

    plt.plot(
        datos["Fecha"],
        datos["S2"],
        label="Pronósticos S2",
        linewidth=1.8,
        marker="o",
        markersize=3
    )

    plt.title(
        "Ventas reales frente a pronósticos SARIMA S2\n"
        "Octubre - diciembre de 2011"
    )

    plt.xlabel("Fecha")
    plt.ylabel("Ventas netas (£)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    ruta = CARPETA_RESULTADOS / "pronosticos_s2_prueba.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGráfica guardada en: {ruta}")

def construir_media_centrada(serie):
    # Calcula la media de siete días calendario usando toda la serie.
    ventas = serie["Venta_neta"].asfreq("D")

    media = ventas.rolling(
        window=7,
        center=True,
        min_periods=1
    ).mean()

    return media

def pronosticar_m2_fuga(historial, fechas_objetivo, media_centrada):
    # Utiliza los rezagos originales de M2.
    ventas = historial["Venta_neta"].dropna().astype(float)

    X = construir_variables_m2(ventas, ventas.index)

    # Agrega una variable calculada con datos de toda la serie.
    X["Media_centrada_7"] = media_centrada.reindex(
        ventas.index
    )

    datos = X.copy()
    datos["Venta_neta"] = ventas
    datos = datos.dropna()

    modelo = sm.OLS(
        datos["Venta_neta"],
        datos.drop(columns=["Venta_neta"])
    ).fit()

    X_futuro = construir_variables_m2(
        ventas,
        fechas_objetivo
    )

    X_futuro["Media_centrada_7"] = media_centrada.reindex(
        fechas_objetivo
    )

    if X_futuro.isna().any().any():
        raise ValueError("Faltan variables para M2 con fuga.")

    predicciones = modelo.predict(X_futuro)

    return pd.Series(
        np.asarray(predicciones),
        index=fechas_objetivo,
        name="M2_fuga"
    )

def probar_fuga_informacion():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie, entrenamiento, prueba = preparar_datos()

    # La media se calcula antes de cualquier separación temporal.
    media_centrada = construir_media_centrada(serie)

    origen = pd.Timestamp("2011-10-02")
    historial = serie.loc[:origen]

    fechas_objetivo = prueba.loc[
        "2011-10-03":"2011-10-09"
    ].index

    reales = prueba.loc[
        fechas_objetivo, "Venta_neta"
    ]

    pred_m2 = pronosticar_m2(
        historial,
        fechas_objetivo
    )

    pred_fuga = pronosticar_m2_fuga(
        historial,
        fechas_objetivo,
        media_centrada
    )

    resultados = pd.DataFrame({
        "Venta_real": reales,
        "M2": pred_m2,
        "M2_fuga": pred_fuga
    })

    print("\nEjercicio 5.4: Demostración de fuga de información")
    print(resultados.round(2).to_string())

    for nombre in ["M2", "M2_fuga"]:
        mae, rmse = calcular_metricas(
            resultados["Venta_real"],
            resultados[nombre]
        )

        print(f"\n{nombre}:")
        print(f"MAE: £{mae:,.2f}")
        print(f"RMSE: £{rmse:,.2f}")

    resultados.to_csv(
        CARPETA_RESULTADOS / "prueba_fuga_informacion.csv",
        encoding="utf-8-sig"
    )

def validar_fuga_informacion():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie, entrenamiento, prueba = preparar_datos()

    # Se calcula sobre toda la serie antes de dividir los datos.
    media_centrada = construir_media_centrada(serie)

    # Validación cruzada aleatoria: cinco folds y diez semillas.
    resultados_aleatorios = []

    for semilla in range(10):
        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=semilla
        )

        reales_semilla = []
        predicciones_semilla = []

        for indices_train, indices_val in kfold.split(entrenamiento):
            historial = entrenamiento.iloc[indices_train].sort_index()
            validacion = entrenamiento.iloc[indices_val].sort_index()

            # Solo se evalúan fechas con rezagos disponibles.
            variables = construir_variables_m2(
                historial["Venta_neta"],
                validacion.index
            )

            fechas_validas = variables.dropna().index

            if len(fechas_validas) == 0:
                continue

            pronosticos = pronosticar_m2_fuga(
                historial,
                fechas_validas,
                media_centrada
            )

            reales_semilla.extend(
                validacion.loc[fechas_validas, "Venta_neta"].to_numpy()
            )
            predicciones_semilla.extend(pronosticos.to_numpy())

        mae, rmse = calcular_metricas(
            reales_semilla,
            predicciones_semilla
        )

        resultados_aleatorios.append({
            "Semilla": semilla,
            "Observaciones": len(reales_semilla),
            "MAE": mae,
            "RMSE": rmse
        })

    resultados_aleatorios = pd.DataFrame(resultados_aleatorios)

    print("\nEjercicio 5.4(a): Validación aleatoria con fuga")
    print(
        resultados_aleatorios[["MAE", "RMSE"]]
        .agg(["mean", "std"])
        .round(2)
        .to_string()
    )

    # Validación temporal: domingos de abril a septiembre.
    inicio = pd.Timestamp("2011-05-01")
    fin = pd.Timestamp("2011-09-30")

    primer_origen = inicio - pd.Timedelta(
        days=inicio.dayofweek + 1
    )

    origenes = pd.date_range(
        primer_origen,
        fin,
        freq="W-SUN"
    )

    registros = []

    for origen in origenes:
        historial = serie.loc[:origen]

        inicio_semana = max(
            origen + pd.Timedelta(days=1),
            inicio
        )

        fin_semana = min(
            origen + pd.Timedelta(days=7),
            fin
        )

        fechas_objetivo = serie.loc[
            inicio_semana:fin_semana
        ].dropna(subset=["Venta_neta"]).index

        if len(fechas_objetivo) == 0:
            continue

        pronosticos = pronosticar_m2_fuga(
            historial,
            fechas_objetivo,
            media_centrada
        )

        for fecha in fechas_objetivo:
            registros.append({
                "Origen": origen,
                "Fecha": fecha,
                "Venta_real": serie.loc[fecha, "Venta_neta"],
                "M2_fuga": pronosticos.loc[fecha]
            })

    detalle_temporal = pd.DataFrame(registros)

    mae, rmse = calcular_metricas(
        detalle_temporal["Venta_real"],
        detalle_temporal["M2_fuga"]
    )

    print("\nEjercicio 5.4(b): Validación temporal con fuga")
    print(f"Observaciones: {len(detalle_temporal)}")
    print(f"MAE: £{mae:,.2f}")
    print(f"RMSE: £{rmse:,.2f}")

    resultados_aleatorios.to_csv(
        CARPETA_RESULTADOS / "validacion_fuga_aleatoria.csv",
        index=False,
        encoding="utf-8-sig"
    )

    detalle_temporal.to_csv(
        CARPETA_RESULTADOS / "validacion_fuga_temporal.csv",
        index=False,
        encoding="utf-8-sig"
    )

def comparar_errores_entrenamiento_prueba():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Carga los resultados obtenidos en el ejercicio 5.2.
    entrenamiento = pd.read_csv(
        CARPETA_RESULTADOS / "validacion_temporal_entrenamiento_resumen.csv"
    )

    prueba = pd.read_csv(
        CARPETA_RESULTADOS / "validacion_temporal_prueba_resumen.csv"
    )

    # Combina las métricas de ambos períodos.
    comparacion = entrenamiento[
        ["Modelo", "MAE", "RMSE"]
    ].merge(
        prueba[["Modelo", "MAE", "RMSE"]],
        on="Modelo",
        suffixes=("_entrenamiento", "_prueba")
    )

    # Calcula cuánto aumentó o disminuyó cada error.
    for metrica in ["MAE", "RMSE"]:
        comparacion[f"Diferencia_{metrica}"] = (
            comparacion[f"{metrica}_prueba"]
            - comparacion[f"{metrica}_entrenamiento"]
        )

        comparacion[f"Cambio_{metrica}_porcentaje"] = (
            comparacion[f"Diferencia_{metrica}"]
            / comparacion[f"{metrica}_entrenamiento"]
        ) * 100

    print("\nEjercicio 5.5: Comparación de errores")
    print(comparacion.round(2).to_string(index=False))

    # Gráfica comparativa del MAE.
    posiciones = np.arange(len(comparacion))
    ancho = 0.35

    plt.figure(figsize=(11, 6))

    plt.bar(
        posiciones - ancho / 2,
        comparacion["MAE_entrenamiento"],
        width=ancho,
        label="Validación temporal (entrenamiento)"
    )

    plt.bar(
        posiciones + ancho / 2,
        comparacion["MAE_prueba"],
        width=ancho,
        label="Validación temporal (prueba)"
    )

    plt.xticks(posiciones, comparacion["Modelo"])
    plt.ylabel("MAE (£)")
    plt.xlabel("Modelo")
    plt.title("Comparación del MAE entre entrenamiento y prueba")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ruta_grafica = (
        CARPETA_RESULTADOS / "comparacion_mae_entrenamiento_prueba.png"
    )

    plt.savefig(ruta_grafica, dpi=300, bbox_inches="tight")
    plt.show()

    # Guarda los resultados para el informe.
    comparacion.to_csv(
        CARPETA_RESULTADOS / "comparacion_errores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nGráfica guardada en: {ruta_grafica}")

    return comparacion

if __name__ == "__main__":
    probar_seis_modelos()
    
    validar_aleatoriamente()
    
    validar_origen_movil(
        fecha_inicio="2011-05-01",
        fecha_fin="2011-09-30",
        nombre="temporal_entrenamiento"
    )
    
    validar_origen_movil(
        fecha_inicio="2011-10-03",
        fecha_fin="2011-12-08",
        nombre="temporal_prueba"
    )
    
    graficar_mejor_modelo()
    
    probar_fuga_informacion()
    
    validar_fuga_informacion()

    comparar_errores_entrenamiento_prueba()