import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import chi2

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO_SERIE = CARPETA_PROYECTO / "Parte 2" / "Resultados" / "serie_diaria_sin_ultimo_dia.csv"
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"

def cargar_serie():
    if not ARCHIVO_SERIE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO_SERIE}")

    serie = pd.read_csv(
        ARCHIVO_SERIE,
        parse_dates=["Fecha"],
        index_col="Fecha"
    )

    return serie.sort_index()

def calcular_media_movil(serie):
    resultado = serie.copy()

    resultado["Media_movil_7"] = (
        resultado["Venta_neta"]
        .rolling(window=7, center=True, min_periods=1)
        .mean()
    )

    return resultado

def graficar_tendencia(serie):
    plt.figure(figsize=(14, 6))

    plt.plot(
        serie.index,
        serie["Venta_neta"],
        color="lightgray",
        linewidth=1,
        label="Venta neta diaria"
    )

    plt.plot(
        serie.index,
        serie["Media_movil_7"],
        color="royalblue",
        linewidth=2,
        label="Media móvil centrada de 7 días"
    )

    plt.title("Ventas netas diarias y media móvil centrada de 7 días")
    plt.xlabel("Fecha")
    plt.ylabel("Venta neta (GBP)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "media_movil_7_dias.png",
        dpi=300
    )

    plt.show()


def calcular_indices_estacionales(serie):
    dias_semana = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo"
    }

    # Conserva únicamente los días con ventas observadas.
    ventas_observadas = serie.dropna(subset=["Venta_neta"]).copy()
    ventas_observadas["Dia_semana"] = ventas_observadas.index.dayofweek

    promedio_general = ventas_observadas["Venta_neta"].mean()

    # Calcula el promedio de ventas para cada día de la semana.
    indices = (
        ventas_observadas.groupby("Dia_semana")["Venta_neta"]
        .agg(Promedio="mean", Dias_observados="count")
        .reindex(range(7))
    )

    indices["Indice_estacional"] = indices["Promedio"] / promedio_general
    indices["Diferencia_porcentual"] = (indices["Indice_estacional"] - 1) * 100
    indices["Dia"] = indices.index.map(dias_semana)

    return indices, promedio_general

def graficar_indices_estacionales(indices):
    datos_grafica = indices.dropna(subset=["Indice_estacional"])

    plt.figure(figsize=(10, 6))

    plt.bar(
        datos_grafica["Dia"],
        datos_grafica["Indice_estacional"],
        color="steelblue"
    )

    plt.axhline(
        y=1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Promedio general"
    )

    plt.title("Índices estacionales de ventas por día de la semana")
    plt.xlabel("Día de la semana")
    plt.ylabel("Índice estacional")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "indices_estacionales_semana.png",
        dpi=300
    )

    plt.show()

def analizar_estacionalidad():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie = cargar_serie()
    indices, promedio_general = calcular_indices_estacionales(serie)

    print("\nEjercicio 3.2: Índices estacionales por día de la semana")
    print(f"Promedio general de ventas: £{promedio_general:,.2f}")

    print("\nResultados por día:")
    print(
        indices[
            ["Dia", "Dias_observados", "Promedio",
             "Indice_estacional", "Diferencia_porcentual"]
        ].to_string(index=False)
    )

    indices.to_csv(
        CARPETA_RESULTADOS / "indices_estacionales_semana.csv",
        encoding="utf-8-sig"
    )

    graficar_indices_estacionales(indices)

def analizar_tendencia():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie = cargar_serie()
    resultado = calcular_media_movil(serie)

    print("\nEjercicio 3.1: Media móvil centrada de 7 días")
    print(f"Primera fecha: {resultado.index.min().date()}")
    print(f"Última fecha: {resultado.index.max().date()}")
    print(f"Días calendario: {len(resultado)}")
    print(f"Días con ventas observadas: {resultado['Venta_neta'].notna().sum()}")
    print(f"Días sin transacciones: {resultado['Venta_neta'].isna().sum()}")

    print("\nPrimeros diez registros:")
    print(resultado[["Venta_neta", "Media_movil_7"]].head(10).to_string())

    print("\nÚltimos diez registros:")
    print(resultado[["Venta_neta", "Media_movil_7"]].tail(10).to_string())

    resultado.to_csv(
        CARPETA_RESULTADOS / "serie_con_media_movil.csv",
        encoding="utf-8-sig"
    )

    graficar_tendencia(resultado)

def calcular_acf_pares(serie, max_rezagos=21):
    ventas = serie["Venta_neta"].to_numpy(dtype=float)
    media = np.nanmean(ventas)
    varianza = np.nanmean((ventas - media) ** 2)

    resultados = []

    for rezago in range(1, max_rezagos + 1):
        anteriores = ventas[:-rezago]
        actuales = ventas[rezago:]

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

def graficar_acf(resultados, cantidad_observaciones):
    banda = 1.96 / np.sqrt(cantidad_observaciones)

    plt.figure(figsize=(12, 6))
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(
        banda, color="red", linestyle="--",
        label="Bandas de referencia del 95 %"
    )
    plt.axhline(-banda, color="red", linestyle="--")

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
    plt.title("Autocorrelación de ventas netas diarias: 21 rezagos")
    plt.xlabel("Rezago (días calendario)")
    plt.ylabel("Autocorrelación")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CARPETA_RESULTADOS / "acf_21_rezagos.png",
        dpi=300
    )
    plt.show()

    return banda

def calcular_ljung_box_exploratorio(resultados, n):
    resultados = resultados.copy()

    resultados["Termino_LB"] = (
        resultados["ACF"] ** 2 /
        (n - resultados["Rezago"])
    )

    resultados["Q_acumulado"] = (
        n * (n + 2) * resultados["Termino_LB"].cumsum()
    )

    resultados["Valor_p_referencia"] = chi2.sf(
        resultados["Q_acumulado"],
        resultados["Rezago"]
    )

    return resultados

def analizar_autocorrelacion():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    serie = cargar_serie()
    n = serie["Venta_neta"].notna().sum()

    resultados = calcular_acf_pares(serie, max_rezagos=21)
    resultados = calcular_ljung_box_exploratorio(resultados, n)

    banda = graficar_acf(resultados, n)

    print("\nEjercicio 3.3: ACF y Ljung-Box")
    print(f"Observaciones con ventas: {n}")
    print(f"Bandas de referencia: +/- {banda:.4f}")

    print("\nAutocorrelaciones por rezago:")
    print(
        resultados[
            ["Rezago", "Pares_validos", "ACF"]
        ].to_string(index=False)
    )

    print("\nEstadísticos Ljung-Box exploratorios:")
    print(
        resultados.loc[
            resultados["Rezago"].isin([7, 14, 21]),
            ["Rezago", "Q_acumulado", "Valor_p_referencia"]
        ].to_string(index=False)
    )

    resultados.to_csv(
        CARPETA_RESULTADOS / "acf_ljung_box.csv",
        index=False,
        encoding="utf-8-sig"
    )

if __name__ == "__main__":
    analizar_tendencia()
    analizar_estacionalidad()
    analizar_autocorrelacion()