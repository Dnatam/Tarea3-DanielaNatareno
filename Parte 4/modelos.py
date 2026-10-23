import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import chi2
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO_SERIE = CARPETA_PROYECTO / "Parte 2" / "Resultados" / "serie_diaria_sin_ultimo_dia.csv"
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"

FECHA_FIN_ENTRENAMIENTO = "2011-09-30"

def cargar_serie():
    if not ARCHIVO_SERIE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO_SERIE}")

    serie = pd.read_csv(
        ARCHIVO_SERIE,
        parse_dates=["Fecha"],
        index_col="Fecha"
    )

    return serie.sort_index()

def calcular_acf_pares(serie, max_rezagos=21):
    valores = serie.to_numpy(dtype=float)
    media = np.nanmean(valores)
    varianza = np.nanmean((valores - media) ** 2)

    resultados = []

    for rezago in range(1, max_rezagos + 1):
        anteriores = valores[:-rezago]
        actuales = valores[rezago:]

        validos = np.isfinite(anteriores) & np.isfinite(actuales)
        cantidad_pares = validos.sum()

        if cantidad_pares > 0 and varianza > 0:
            covarianza = np.mean(
                (anteriores[validos] - media) *
                (actuales[validos] - media)
            )
            autocorrelacion = covarianza / varianza
        else:
            autocorrelacion = np.nan

        resultados.append({
            "Rezago": rezago,
            "Pares_validos": cantidad_pares,
            "ACF": autocorrelacion
        })

    return pd.DataFrame(resultados)

def ajustar_ols(serie):
    entrenamiento = serie.loc[
        :FECHA_FIN_ENTRENAMIENTO
    ].copy()

    # Mantiene únicamente los días con ventas observadas.
    datos_ols = entrenamiento.dropna(
        subset=["Venta_neta"]
    ).copy()

    datos_ols["Dia_semana"] = datos_ols.index.dayofweek

    # Una variable indicadora por cada día abierto, sin intercepto.
    variables = pd.get_dummies(
        datos_ols["Dia_semana"],
        prefix="Dia",
        dtype=float
    )

    ventas = datos_ols["Venta_neta"].astype(float)

    modelo = sm.OLS(ventas, variables).fit()

    datos_ols["Venta_estimada"] = modelo.fittedvalues
    datos_ols["Residuo"] = modelo.resid

    # Recupera el calendario para conservar los rezagos reales.
    residuos_calendario = entrenamiento["Venta_neta"].copy()
    residuos_calendario[:] = np.nan
    residuos_calendario.loc[datos_ols.index] = datos_ols["Residuo"]

    return modelo, datos_ols, residuos_calendario

def graficar_acf_residuos(resultados, n):
    banda = 1.96 / np.sqrt(n)

    plt.figure(figsize=(12, 6))
    plt.axhline(0, color="black", linewidth=1)

    plt.axhline(
        banda,
        color="red",
        linestyle="--",
        label="Bandas de referencia del 95 %"
    )

    plt.axhline(
        -banda,
        color="red",
        linestyle="--"
    )

    plt.vlines(
        resultados["Rezago"],
        0,
        resultados["ACF"],
        color="steelblue",
        linewidth=2
    )

    plt.scatter(
        resultados["Rezago"],
        resultados["ACF"],
        color="steelblue",
        s=25
    )

    plt.xticks(range(1, 22))
    plt.title("ACF de residuos de regresión OLS por día de la semana")
    plt.xlabel("Rezago (días calendario)")
    plt.ylabel("Autocorrelación")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "acf_residuos_ols.png",
        dpi=300
    )

    plt.show()

    return banda

def analizar_ols():
    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    serie = cargar_serie()
    modelo, datos_ols, residuos = ajustar_ols(serie)

    acf = calcular_acf_pares(
        residuos,
        max_rezagos=21
    )

    banda = graficar_acf_residuos(
        acf,
        len(datos_ols)
    )

    print("\nEjercicio 4.1: Regresión OLS por día de la semana")
    print(f"Inicio del entrenamiento: {datos_ols.index.min().date()}")
    print(f"Fin del entrenamiento: {FECHA_FIN_ENTRENAMIENTO}")
    print(f"Observaciones utilizadas: {len(datos_ols)}")
    print(f"R cuadrado: {modelo.rsquared:.4f}")
    print(f"R cuadrado ajustado: {modelo.rsquared_adj:.4f}")

    print("\nCoeficientes estimados:")
    print(modelo.params.to_string())

    print("\nResumen estadístico del modelo:")
    print(modelo.summary())

    print("\nACF de residuos:")
    print(acf.to_string(index=False))
    print(f"\nBandas de referencia: +/- {banda:.4f}")

    datos_ols.to_csv(
        CARPETA_RESULTADOS / "resultados_ols.csv",
        encoding="utf-8-sig"
    )

    acf.to_csv(
        CARPETA_RESULTADOS / "acf_residuos_ols.csv",
        index=False,
        encoding="utf-8-sig"
    )

    residuos.to_csv(
        CARPETA_RESULTADOS / "residuos_ols_calendario.csv",
        encoding="utf-8-sig",
        header=["Residuo"]
    )

def ajustar_ar1_residuos(residuos_ols):
    datos = pd.DataFrame({
        "Residuo_OLS": residuos_ols,
        "Residuo_anterior": residuos_ols.shift(1)
    })

    # Conserva pares de días calendario consecutivos con datos.
    pares_validos = datos.dropna().copy()

    X = sm.add_constant(pares_validos["Residuo_anterior"])
    y = pares_validos["Residuo_OLS"]

    modelo = sm.OLS(y, X).fit()

    pares_validos["Residuo_estimado"] = modelo.fittedvalues
    pares_validos["Innovacion_AR1"] = modelo.resid

    # Conserva los días sin innovación estimable como NaN.
    innovaciones = pd.Series(
        np.nan,
        index=residuos_ols.index,
        name="Innovacion_AR1"
    )

    innovaciones.loc[pares_validos.index] = modelo.resid

    return modelo, pares_validos, innovaciones

def calcular_ljung_box_ar1(acf, n):
    resultados = acf.copy()

    resultados["Termino_LB"] = (
        resultados["ACF"] ** 2 /
        (n - resultados["Rezago"])
    )

    resultados["Q_acumulado"] = (
        n * (n + 2) * resultados["Termino_LB"].cumsum()
    )

    # Se descuenta un parámetro autorregresivo estimado.
    resultados["Grados_libertad"] = resultados["Rezago"] - 1

    resultados["Valor_p_referencia"] = np.where(
        resultados["Grados_libertad"] > 0,
        chi2.sf(
            resultados["Q_acumulado"],
            resultados["Grados_libertad"].clip(lower=1)
        ),
        np.nan
    )

    return resultados

def graficar_acf_ar1(acf, n):
    banda = 1.96 / np.sqrt(n)

    plt.figure(figsize=(12, 6))
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(
        banda,
        color="red",
        linestyle="--",
        label="Bandas de referencia del 95 %"
    )
    plt.axhline(-banda, color="red", linestyle="--")

    plt.vlines(
        acf["Rezago"],
        0,
        acf["ACF"],
        color="steelblue",
        linewidth=2
    )
    plt.scatter(
        acf["Rezago"],
        acf["ACF"],
        color="steelblue",
        s=25
    )

    plt.xticks(range(1, 22))
    plt.title("ACF de innovaciones del modelo AR(1)")
    plt.xlabel("Rezago (días calendario)")
    plt.ylabel("Autocorrelación")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "acf_residuos_ar1.png",
        dpi=300
    )
    plt.show()

    return banda

def analizar_ar1():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie = cargar_serie()
    modelo_ols, datos_ols, residuos_ols = ajustar_ols(serie)

    modelo_ar1, pares, innovaciones = ajustar_ar1_residuos(residuos_ols)

    n = innovaciones.notna().sum()
    acf = calcular_acf_pares(innovaciones, max_rezagos=21)
    diagnostico = calcular_ljung_box_ar1(acf, n)
    banda = graficar_acf_ar1(acf, n)

    print("\nEjercicio 4.2: AR(1) sobre residuos OLS")
    print(f"Pares consecutivos utilizados: {len(pares)}")
    print(f"Coeficiente AR(1): {modelo_ar1.params['Residuo_anterior']:.6f}")
    print(f"Intercepto: {modelo_ar1.params['const']:.6f}")
    print(f"R cuadrado: {modelo_ar1.rsquared:.4f}")

    print("\nResumen del modelo AR(1):")
    print(modelo_ar1.summary())

    print("\nACF de innovaciones:")
    print(
        diagnostico[
            ["Rezago", "Pares_validos", "ACF"]
        ].to_string(index=False)
    )

    print("\nDiagnóstico Ljung-Box en rezagos 6 y 12:")
    print(
        diagnostico.loc[
            diagnostico["Rezago"].isin([6, 12]),
            ["Rezago", "Q_acumulado",
             "Grados_libertad", "Valor_p_referencia"]
        ].to_string(index=False)
    )

    print(f"\nBandas de referencia: +/- {banda:.4f}")

    pares.to_csv(
        CARPETA_RESULTADOS / "resultados_ar1.csv",
        encoding="utf-8-sig"
    )

    innovaciones.to_csv(
        CARPETA_RESULTADOS / "innovaciones_ar1_calendario.csv",
        encoding="utf-8-sig"
    )

    diagnostico.to_csv(
        CARPETA_RESULTADOS / "diagnostico_ar1.csv",
        index=False,
        encoding="utf-8-sig"
    )

def preparar_datos_sarima():
    serie = cargar_serie()

    entrenamiento = serie.loc[: "2011-09-30"].copy()
    dias_abiertos = entrenamiento.dropna(subset=["Venta_neta"]).copy()

    # El logaritmo requiere valores estrictamente positivos.
    ventas_no_positivas = dias_abiertos["Venta_neta"] <= 0

    if ventas_no_positivas.any():
        fechas = dias_abiertos.index[ventas_no_positivas]
        raise ValueError(
            f"Hay {len(fechas)} días con ventas no positivas: "
            f"{fechas.strftime('%Y-%m-%d').tolist()}"
        )

    dias_abiertos["Log_venta"] = np.log(dias_abiertos["Venta_neta"])

    return dias_abiertos


def ajustar_modelo_sarima(valores, nombre, orden, orden_estacional, tendencia):
    print(f"\nAjustando modelo {nombre}...")

    modelo = SARIMAX(
        valores,
        order=orden,
        seasonal_order=orden_estacional,
        trend=tendencia,
        enforce_stationarity=True,
        enforce_invertibility=True
    )

    resultado = modelo.fit(disp=False, maxiter=500)

    convergio = resultado.mle_retvals.get("converged", False)

    print(f"Convergió: {convergio}")
    print(f"Iteraciones: {resultado.mle_retvals.get('iterations', 'No disponible')}")
    print(f"AIC: {resultado.aic:.4f}")
    print(f"BIC: {resultado.bic:.4f}")

    print("\nParámetros estimados:")
    print(resultado.params)

    if not convergio:
        print(f"Advertencia: {nombre} no convergió.")

    return resultado

def analizar_sarima():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    datos = preparar_datos_sarima()
    valores = datos["Log_venta"].values

    print("\nEjercicio 4.3: Modelos SARIMA S1 y S2")
    print(f"Inicio del entrenamiento: {datos.index.min().date()}")
    print(f"Fin del entrenamiento: {datos.index.max().date()}")
    print(f"Observaciones utilizadas: {len(datos)}")
    print(f"Venta mínima: £{datos['Venta_neta'].min():,.2f}")
    print(f"Venta máxima: £{datos['Venta_neta'].max():,.2f}")

    modelo_s1 = ajustar_modelo_sarima(
        valores,
        "S1",
        orden=(1, 0, 0),
        orden_estacional=(1, 0, 0, 6),
        tendencia="c"
    )

    modelo_s2 = ajustar_modelo_sarima(
        valores,
        "S2",
        orden=(0, 1, 1),
        orden_estacional=(0, 1, 1, 6),
        tendencia="n"
    )

    comparacion = pd.DataFrame([
        {
            "Modelo": "S1",
            "Orden": "(1,0,0)(1,0,0,6)",
            "Convergencia": modelo_s1.mle_retvals.get("converged", False),
            "AIC": modelo_s1.aic,
            "BIC": modelo_s1.bic
        },
        {
            "Modelo": "S2",
            "Orden": "(0,1,1)(0,1,1,6)",
            "Convergencia": modelo_s2.mle_retvals.get("converged", False),
            "AIC": modelo_s2.aic,
            "BIC": modelo_s2.bic
        }
    ])

    print("\nComparación de modelos:")
    print(comparacion.to_string(index=False))

    comparacion.to_csv(
        CARPETA_RESULTADOS / "comparacion_sarima.csv",
        index=False,
        encoding="utf-8-sig"
    )

    residuos = pd.DataFrame({
        "Fecha": datos.index,
        "Residuo_S1": modelo_s1.resid,
        "Residuo_S2": modelo_s2.resid
    })

    residuos.to_csv(
        CARPETA_RESULTADOS / "residuos_sarima.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return modelo_s1, modelo_s2

def graficar_acf_sarima(residuos, nombre):
    cantidad = len(residuos)
    max_rezagos = 21

    autocorrelaciones = [
        residuos.autocorr(lag=rezago)
        for rezago in range(1, max_rezagos + 1)
    ]

    banda = 1.96 / np.sqrt(cantidad)
    rezagos = np.arange(1, max_rezagos + 1)

    plt.figure(figsize=(12, 6))
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(
        banda,
        color="red",
        linestyle="--",
        label="Bandas de referencia del 95 %"
    )
    plt.axhline(-banda, color="red", linestyle="--")

    plt.vlines(
        rezagos, 0, autocorrelaciones,
        color="steelblue", linewidth=2
    )
    plt.scatter(
        rezagos, autocorrelaciones,
        color="steelblue", s=25
    )

    plt.xticks(rezagos)
    plt.title(f"ACF de residuos del modelo SARIMA {nombre}")
    plt.xlabel("Rezago (días abiertos)")
    plt.ylabel("Autocorrelación")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / f"acf_residuos_sarima_{nombre.lower()}.png",
        dpi=300
    )
    plt.show()

    return pd.DataFrame({
        "Rezago": rezagos,
        "ACF": autocorrelaciones
    }), banda

def diagnosticar_modelo_sarima(residuos, nombre, parametros):
    residuos = pd.Series(residuos, dtype=float).dropna()
    resultados_acf, banda = graficar_acf_sarima(residuos, nombre)

    # Ajusta los grados de libertad según los parámetros AR y MA.
    diagnostico = acorr_ljungbox(
        residuos,
        lags=[6, 12],
        model_df=parametros,
        return_df=True
    )

    diagnostico = diagnostico.reset_index()
    diagnostico.columns = ["Rezago", "Q_Ljung_Box", "Valor_p"]

    diagnostico["Grados_libertad"] = (
        diagnostico["Rezago"] - parametros
    )

    print(f"\nDiagnóstico de residuos del modelo {nombre}")
    print(f"Residuos utilizados: {len(residuos)}")
    print(f"Media de residuos: {residuos.mean():.6f}")
    print(f"Bandas de referencia: +/- {banda:.4f}")

    print("\nACF de residuos:")
    print(resultados_acf.to_string(index=False))

    print("\nPrueba de Ljung-Box:")
    print(
        diagnostico[
            ["Rezago", "Q_Ljung_Box", "Grados_libertad", "Valor_p"]
        ].to_string(index=False)
    )

    resultados_acf.to_csv(
        CARPETA_RESULTADOS / f"acf_sarima_{nombre.lower()}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    diagnostico.to_csv(
        CARPETA_RESULTADOS / f"ljung_box_sarima_{nombre.lower()}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return diagnostico

def analizar_diagnostico_sarima():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    archivo_residuos = CARPETA_RESULTADOS / "residuos_sarima.csv"

    if not archivo_residuos.exists():
        raise FileNotFoundError(
            "Primero debes ejecutar el ejercicio 4.3."
        )

    datos = pd.read_csv(archivo_residuos)

    residuos_s1 = datos["Residuo_S1"]

    # Se descartan los primeros siete residuos de S2.
    residuos_s2 = datos["Residuo_S2"].iloc[7:]

    print("\nEjercicio 4.4: Diagnóstico de residuos SARIMA")

    diagnostico_s1 = diagnosticar_modelo_sarima(
        residuos_s1, "S1", parametros=2
    )

    diagnostico_s2 = diagnosticar_modelo_sarima(
        residuos_s2, "S2", parametros=2
    )

    return diagnostico_s1, diagnostico_s2

def analizar_pronosticos_largo_plazo(modelo_s1, modelo_s2):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    horizontes = 60

    pronostico_s1 = modelo_s1.get_forecast(steps=horizontes)
    pronostico_s2 = modelo_s2.get_forecast(steps=horizontes)

    media_s1 = pronostico_s1.predicted_mean
    media_s2 = pronostico_s2.predicted_mean

    intervalo_s1 = pronostico_s1.conf_int(alpha=0.05)
    intervalo_s2 = pronostico_s2.conf_int(alpha=0.05)

    resultados = pd.DataFrame({
        "Horizonte": range(1, horizontes + 1),
        "Pronostico_S1": np.asarray(media_s1),
        "Limite_inferior_S1": np.asarray(intervalo_s1)[:, 0],
        "Limite_superior_S1": np.asarray(intervalo_s1)[:, 1],
        "Pronostico_S2": np.asarray(media_s2),
        "Limite_inferior_S2": np.asarray(intervalo_s2)[:, 0],
        "Limite_superior_S2": np.asarray(intervalo_s2)[:, 1]
    })

    resultados.to_csv(
        CARPETA_RESULTADOS / "pronosticos_largo_plazo.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(13, 6))

    plt.plot(
        resultados["Horizonte"],
        resultados["Pronostico_S1"],
        label="SARIMA S1",
        color="royalblue"
    )

    plt.plot(
        resultados["Horizonte"],
        resultados["Pronostico_S2"],
        label="SARIMA S2",
        color="darkorange"
    )

    plt.fill_between(
        resultados["Horizonte"],
        resultados["Limite_inferior_S1"],
        resultados["Limite_superior_S1"],
        color="royalblue",
        alpha=0.15
    )

    plt.fill_between(
        resultados["Horizonte"],
        resultados["Limite_inferior_S2"],
        resultados["Limite_superior_S2"],
        color="darkorange",
        alpha=0.15
    )

    plt.title("Pronósticos a largo plazo: SARIMA S1 y S2")
    plt.xlabel("Horizonte (días abiertos)")
    plt.ylabel("Pronóstico en la escala utilizada para ajustar los modelos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "pronosticos_largo_plazo.png",
        dpi=300
    )

    plt.show()

    print("\nEjercicio 4.5: Pronósticos a largo plazo")
    print("\nPronósticos seleccionados:")

    print(
        resultados.loc[
            resultados["Horizonte"].isin([1, 7, 14, 30, 60]),
            ["Horizonte", "Pronostico_S1", "Pronostico_S2"]
        ].to_string(index=False)
    )

    print("\nAmplitud de los intervalos al horizonte 60:")

    for nombre in ["S1", "S2"]:
        amplitud = (
            resultados[f"Limite_superior_{nombre}"].iloc[-1]
            - resultados[f"Limite_inferior_{nombre}"].iloc[-1]
        )
        print(f"{nombre}: {amplitud:.6f}")

if __name__ == "__main__":
    analizar_ols()
    analizar_ar1()
    modelo_s1, modelo_s2 = analizar_sarima()
    analizar_diagnostico_sarima()
    analizar_pronosticos_largo_plazo(modelo_s1, modelo_s2)