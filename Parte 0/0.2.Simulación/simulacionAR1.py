import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Parámetros de la simulación
N_OBSERVACIONES = 300
N_DESCARTE = 200
N_REPLICAS = 5000
VALORES_PHI = [0.0, 0.5, 0.9, 1.0]
SEMILLAS = [8103, 8104, 8105, 8106]
CARPETA_RESULTADOS = Path("ResultadosAR1")
CARPETA_RESULTADOS.mkdir(exist_ok=True)

# Simula un proceso AR(1) y descarta las observaciones iniciales.
def simular_ar1(phi, n_observaciones, n_descarte, generador):
    total = n_observaciones + n_descarte
    errores = generador.normal(loc=0, scale=1, size=total)
    serie = np.zeros(total + 1)
    for t in range(1, total + 1):
        serie[t] = phi * serie[t - 1] + errores[t - 1]
    return serie[n_descarte + 1:]

# Calcula la autocorrelación muestral usando la media de la serie.
def calcular_autocorrelacion(serie, rezago):
    desviaciones = serie - np.mean(serie)
    numerador = np.sum(desviaciones[rezago:] * desviaciones[:-rezago])
    denominador = np.sum(desviaciones ** 2)
    return numerador / denominador

# Estima el coeficiente AR(1) mediante OLS con intercepto.
def estimar_phi_ols(serie):
    x = serie[:-1]
    y = serie[1:]
    x_centrado = x - np.mean(x)
    y_centrado = y - np.mean(y)
    numerador = np.sum(x_centrado * y_centrado)
    denominador = np.sum(x_centrado ** 2)
    return numerador / denominador

# Calcula el error estándar de un promedio de réplicas independientes.
def calcular_error_estandar(valores):
    return np.std(valores, ddof=1) / np.sqrt(len(valores))

# Genera una realización de cada escenario y guarda la figura.
def graficar_realizaciones():
    figura, ejes = plt.subplots(2, 2, figsize=(13, 8))
    for eje, phi, semilla in zip(ejes.flat, VALORES_PHI, SEMILLAS):
        generador = np.random.default_rng(semilla)
        serie = simular_ar1(phi, N_OBSERVACIONES, N_DESCARTE, generador)
        eje.plot(np.arange(1, N_OBSERVACIONES + 1), serie, linewidth=1)
        eje.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        eje.set_title(f"Proceso AR(1): phi = {phi}")
        eje.set_xlabel("Tiempo")
        eje.set_ylabel("Valor de la serie")
        eje.grid(alpha=0.25)
    figura.suptitle("Realizaciones de procesos AR(1)", fontsize=14)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "realizaciones_ar1.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figura)


# Calcula los estadísticos solicitados para cada valor de phi.
def ejecutar_replicas():
    resultados = []
    umbral = 1.96 / np.sqrt(N_OBSERVACIONES)
    for phi, semilla in zip(VALORES_PHI, SEMILLAS):
        generador = np.random.default_rng(semilla)
        valores_r1 = np.empty(N_REPLICAS)
        valores_r5 = np.empty(N_REPLICAS)
        valores_ols = np.empty(N_REPLICAS)
        rechazos = np.empty(N_REPLICAS)
        for replica in range(N_REPLICAS):
            serie = simular_ar1(phi, N_OBSERVACIONES, N_DESCARTE, generador)
            valores_r1[replica] = calcular_autocorrelacion(serie, 1)
            valores_r5[replica] = calcular_autocorrelacion(serie, 5)
            valores_ols[replica] = estimar_phi_ols(serie)
            rechazos[replica] = float(abs(valores_r1[replica]) > umbral)
        resultados.append({
            "phi": phi,
            "promedio_r1": np.mean(valores_r1),
            "ee_r1": calcular_error_estandar(valores_r1),
            "promedio_r5": np.mean(valores_r5),
            "ee_r5": calcular_error_estandar(valores_r5),
            "phi_elevado_5": phi ** 5,
            "tasa_rechazo": np.mean(rechazos),
            "ee_tasa_rechazo": calcular_error_estandar(rechazos),
            "promedio_phi_ols": np.mean(valores_ols),
            "ee_phi_ols": calcular_error_estandar(valores_ols)
        })
    return pd.DataFrame(resultados)

# Ejecuta las simulaciones y guarda los resultados reproducibles.
def main():
    graficar_realizaciones()
    tabla = ejecutar_replicas()
    tabla.to_csv(CARPETA_RESULTADOS / "tabla_ar1.csv", index=False)
    print("\nResultados de 5000 réplicas por escenario:")
    print(tabla.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print(f"\nUmbral de rechazo: {1.96 / np.sqrt(N_OBSERVACIONES):.6f}")
    print(f"Archivos guardados en: {CARPETA_RESULTADOS.resolve()}")

if __name__ == "__main__":
    main()
