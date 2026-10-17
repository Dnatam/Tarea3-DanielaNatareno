
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox

# Parámetros de las simulaciones y pruebas estadísticas
N_OBSERVACIONES = 300
N_REPLICAS = 5000
N_REZAGOS = 10
NIVEL_SIGNIFICANCIA = 0.05
SEMILLAS = {"ruido_blanco": 8303, "tendencia": 8304, "semanal": 8305}
CARPETA_RESULTADOS = Path("Resultados")
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

def generar_media(tipo, n_observaciones):
    """Define la media del proceso para cada escenario."""
    tiempo = np.arange(1, n_observaciones + 1)
    if tipo == "ruido_blanco":
        return np.zeros(n_observaciones)
    if tipo == "tendencia":
        return 2 * tiempo / n_observaciones
    if tipo == "semanal":
        return np.where(tiempo % 7 == 6, -1.5, 0.25)
    raise ValueError(f"Proceso no reconocido: {tipo}")

def simular_proceso(tipo, n_observaciones, generador):
    """Genera una serie con errores normales independientes y media determinista."""
    media = generar_media(tipo, n_observaciones)
    errores = generador.normal(loc=0, scale=1, size=n_observaciones)
    return media + errores

def calcular_autocorrelacion(serie, rezago):
    """Calcula la autocorrelación muestral con la misma definición de los ejercicios anteriores."""
    desviaciones = serie - np.mean(serie)
    numerador = np.sum(desviaciones[rezago:] * desviaciones[:-rezago])
    denominador = np.sum(desviaciones ** 2)
    return numerador / denominador

def calcular_ljung_box(serie, n_rezagos):
    """Calcula manualmente el estadístico Q y su valor p mediante chi-cuadrado."""
    n = len(serie)
    autocorrelaciones = np.array([
        calcular_autocorrelacion(serie, rezago)
        for rezago in range(1, n_rezagos + 1)
    ])
    rezagos = np.arange(1, n_rezagos + 1)
    estadistico_q = n * (n + 2) * np.sum(autocorrelaciones ** 2 / (n - rezagos))
    valor_p = chi2.sf(estadistico_q, df=n_rezagos)
    return estadistico_q, valor_p

def calcular_error_estandar(valores):
    """Calcula el error estándar de un promedio obtenido mediante réplicas independientes."""
    return np.std(valores, ddof=1) / np.sqrt(len(valores))

def ejecutar_simulaciones():
    """Estima las tasas de rechazo de ambas pruebas en los tres procesos."""
    resultados = []
    umbral_r1 = 1.96 / np.sqrt(N_OBSERVACIONES)
    valor_critico_q = chi2.ppf(1 - NIVEL_SIGNIFICANCIA, df=N_REZAGOS)
    for tipo, semilla in SEMILLAS.items():
        generador = np.random.default_rng(semilla)
        rechazos_r1 = np.empty(N_REPLICAS, dtype=float)
        rechazos_ljung_box = np.empty(N_REPLICAS, dtype=float)
        estadisticos_q = np.empty(N_REPLICAS)
        valores_p = np.empty(N_REPLICAS)
        for replica in range(N_REPLICAS):
            serie = simular_proceso(tipo, N_OBSERVACIONES, generador)
            r1 = calcular_autocorrelacion(serie, 1)
            estadistico_q, valor_p = calcular_ljung_box(serie, N_REZAGOS)
            rechazos_r1[replica] = float(abs(r1) > umbral_r1)
            rechazos_ljung_box[replica] = float(valor_p < NIVEL_SIGNIFICANCIA)
            estadisticos_q[replica] = estadistico_q
            valores_p[replica] = valor_p
        resultados.append({
            "proceso": tipo,
            "tasa_rechazo_r1": np.mean(rechazos_r1),
            "ee_rechazo_r1": calcular_error_estandar(rechazos_r1),
            "tasa_rechazo_ljung_box": np.mean(rechazos_ljung_box),
            "ee_rechazo_ljung_box": calcular_error_estandar(rechazos_ljung_box),
            "promedio_q10": np.mean(estadisticos_q),
            "ee_q10": calcular_error_estandar(estadisticos_q),
            "promedio_valor_p": np.mean(valores_p)
        })
    return pd.DataFrame(resultados), valor_critico_q

def verificar_implementacion():
    """Compara la implementación manual con statsmodels en una misma serie."""
    generador = np.random.default_rng(8403)
    serie = simular_proceso("semanal", N_OBSERVACIONES, generador)
    estadistico_manual, valor_p_manual = calcular_ljung_box(serie, N_REZAGOS)
    resultado_statsmodels = acorr_ljungbox(serie, lags=[N_REZAGOS], return_df=True)
    estadistico_libreria = resultado_statsmodels["lb_stat"].iloc[0]
    valor_p_libreria = resultado_statsmodels["lb_pvalue"].iloc[0]
    print("\nVerificación de la implementación manual:")
    print(f"Q(10) manual: {estadistico_manual:.8f}")
    print(f"Q(10) statsmodels: {estadistico_libreria:.8f}")
    print(f"Valor p manual: {valor_p_manual:.8f}")
    print(f"Valor p statsmodels: {valor_p_libreria:.8f}")
    if not np.allclose([estadistico_manual, valor_p_manual], [estadistico_libreria, valor_p_libreria]):
        raise AssertionError("La implementación manual no coincide con statsmodels.")

def graficar_tasas(tabla):
    """Compara visualmente las tasas de rechazo de las dos pruebas."""
    posiciones = np.arange(len(tabla))
    ancho = 0.35
    figura, eje = plt.subplots(figsize=(10, 6))
    eje.bar(posiciones - ancho / 2, tabla["tasa_rechazo_r1"] * 100, ancho, label="Prueba de rezago 1")
    eje.bar(posiciones + ancho / 2, tabla["tasa_rechazo_ljung_box"] * 100, ancho, label="Ljung-Box Q(10)")
    eje.set_xticks(posiciones)
    eje.set_xticklabels(["Ruido blanco", "Tendencia lineal", "Patrón semanal"])
    eje.set_ylabel("Tasa de rechazo (%)")
    eje.set_ylim(0, 105)
    eje.set_title("Comparación de tasas de rechazo al 5 %")
    eje.legend()
    eje.grid(axis="y", alpha=0.25)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "comparacion_pruebas.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figura)

def main():
    """Ejecuta las pruebas, valida la implementación y guarda los resultados."""
    verificar_implementacion()
    tabla, valor_critico_q = ejecutar_simulaciones()
    tabla.to_csv(CARPETA_RESULTADOS / "tabla_ljung_box.csv", index=False)
    graficar_tasas(tabla)
    print("\nResultados de 5000 réplicas por escenario:")
    print(tabla.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print(f"\nValor crítico de chi-cuadrado con 10 grados de libertad: {valor_critico_q:.6f}")
    print(f"Archivos guardados en: {CARPETA_RESULTADOS.resolve()}")

if __name__ == "__main__":
    main()
