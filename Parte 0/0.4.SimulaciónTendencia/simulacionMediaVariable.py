
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de las simulaciones y archivos de salida
N_OBSERVACIONES = 300
N_REPLICAS = 5000
SEMILLAS = {"tendencia": 8203, "semanal": 8204}
CARPETA_RESULTADOS = Path("Resultados")
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Construye la media determinista de cada proceso sin memoria.
def generar_media(tipo, n_observaciones):
    tiempo = np.arange(1, n_observaciones + 1)
    if tipo == "tendencia":
        return 2 * tiempo / n_observaciones
    if tipo == "semanal":
        return np.where(tiempo % 7 == 6, -1.5, 0.25)
    raise ValueError(f"Tipo de proceso no reconocido: {tipo}")

# Genera observaciones independientes alrededor de una media variable.
def simular_proceso(tipo, n_observaciones, generador):
    media = generar_media(tipo, n_observaciones)
    errores = generador.normal(loc=0, scale=1, size=n_observaciones)
    return media + errores

# Calcula la autocorrelación muestral con una media global.
def calcular_autocorrelacion(serie, rezago):
    desviaciones = serie - np.mean(serie)
    numerador = np.sum(desviaciones[rezago:] * desviaciones[:-rezago])
    denominador = np.sum(desviaciones ** 2)
    return numerador / denominador

# Estima una regresión AR(1) con intercepto como referencia comparativa.
def estimar_phi_ols(serie):
    x = serie[:-1]
    y = serie[1:]
    x_centrado = x - np.mean(x)
    y_centrado = y - np.mean(y)
    return np.sum(x_centrado * y_centrado) / np.sum(x_centrado ** 2)

# Calcula el error estándar de la media de réplicas independientes.
def calcular_error_estandar(valores):
    return np.std(valores, ddof=1) / np.sqrt(len(valores))

# Grafica una realización de cada proceso y su media determinista.
def graficar_realizaciones():
    figura, ejes = plt.subplots(2, 1, figsize=(12, 8))
    tiempo = np.arange(1, N_OBSERVACIONES + 1)
    for eje, tipo, titulo in zip(ejes, SEMILLAS, ["Tendencia lineal", "Patrón semanal"]):
        generador = np.random.default_rng(SEMILLAS[tipo])
        media = generar_media(tipo, N_OBSERVACIONES)
        serie = simular_proceso(tipo, N_OBSERVACIONES, generador)
        eje.plot(tiempo, serie, linewidth=1, label="Serie simulada")
        eje.plot(tiempo, media, linewidth=2, label="Media determinista")
        eje.set_title(titulo)
        eje.set_xlabel("Tiempo")
        eje.set_ylabel("Valor de la serie")
        eje.grid(alpha=0.25)
        eje.legend()
    figura.suptitle("Procesos sin memoria con media variable", fontsize=14)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "realizaciones_media_variable.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figura)

# Resume autocorrelaciones, rechazos y estimaciones OLS de cada escenario.
def ejecutar_replicas():
    resultados = []
    umbral = 1.96 / np.sqrt(N_OBSERVACIONES)
    for tipo, semilla in SEMILLAS.items():
        generador = np.random.default_rng(semilla)
        valores_r1 = np.empty(N_REPLICAS)
        valores_r5 = np.empty(N_REPLICAS)
        valores_r7 = np.empty(N_REPLICAS)
        valores_ols = np.empty(N_REPLICAS)
        rechazos = np.empty(N_REPLICAS)
        for replica in range(N_REPLICAS):
            serie = simular_proceso(tipo, N_OBSERVACIONES, generador)
            valores_r1[replica] = calcular_autocorrelacion(serie, 1)
            valores_r5[replica] = calcular_autocorrelacion(serie, 5)
            valores_r7[replica] = calcular_autocorrelacion(serie, 7)
            valores_ols[replica] = estimar_phi_ols(serie)
            rechazos[replica] = float(abs(valores_r1[replica]) > umbral)
        resultados.append({
            "proceso": tipo,
            "promedio_r1": np.mean(valores_r1),
            "ee_r1": calcular_error_estandar(valores_r1),
            "promedio_r5": np.mean(valores_r5),
            "ee_r5": calcular_error_estandar(valores_r5),
            "promedio_r7": np.mean(valores_r7),
            "ee_r7": calcular_error_estandar(valores_r7),
            "tasa_rechazo": np.mean(rechazos),
            "ee_tasa_rechazo": calcular_error_estandar(rechazos),
            "promedio_phi_ols": np.mean(valores_ols),
            "ee_phi_ols": calcular_error_estandar(valores_ols)
        })
    return pd.DataFrame(resultados)

# Ejecuta el experimento y guarda la figura y los resultados.
def main():
    graficar_realizaciones()
    tabla = ejecutar_replicas()
    tabla.to_csv(CARPETA_RESULTADOS / "tabla_media_variable.csv", index=False)
    print("\nResultados de 5000 réplicas por escenario:")
    print(tabla.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print(f"\nUmbral de rechazo para r1: {1.96 / np.sqrt(N_OBSERVACIONES):.6f}")
    print(f"Archivos guardados en: {CARPETA_RESULTADOS.resolve()}")

if __name__ == "__main__":
    main()
