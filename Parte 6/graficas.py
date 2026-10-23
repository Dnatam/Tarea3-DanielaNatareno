import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"
CARPETA_GRAFICAS = CARPETA_RESULTADOS / "Graficas"

CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

# Carga los resultados obtenidos en clientes.py.
rfm = pd.read_csv(CARPETA_RESULTADOS / "clientes_rfm_recompra.csv")
auc = pd.read_csv(CARPETA_RESULTADOS / "resultados_auc.csv")
nuevos = pd.read_csv(CARPETA_RESULTADOS / "clientes_nuevos_resumen.csv")

# GRÁFICA 1: Distribución de quintiles RFM.
def graficar_quintiles():
    puntajes = ["R_score", "F_score", "M_score"]
    etiquetas = ["Recencia", "Frecuencia", "Monto"]

    x = np.arange(1, 6)
    ancho = 0.25

    plt.figure(figsize=(9, 5))

    for i, columna in enumerate(puntajes):
        conteos = rfm[columna].value_counts().reindex(range(1, 6), fill_value=0)
        plt.bar(
            x + (i - 1) * ancho,
            conteos.values,
            width=ancho,
            label=etiquetas[i]
        )

    plt.title("Distribución de clientes por quintiles RFM")
    plt.xlabel("Puntuación")
    plt.ylabel("Cantidad de clientes")
    plt.xticks(x)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)    
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / "quintiles_rfm.png",
        dpi=300
    )
    plt.close()

# GRÁFICA 2: Tasa de recompra según R, F y M.
def graficar_recompra():
    plt.figure(figsize=(9, 5))

    for columna, etiqueta in [
        ("R_score", "Recencia"),
        ("F_score", "Frecuencia"),
        ("M_score", "Monto")
    ]:
        tasas = rfm.groupby(columna)["compra"].mean() * 100

        plt.plot(
            tasas.index,
            tasas.values,
            marker="o",
            linewidth=2,
            label=etiqueta
        )

    plt.title("Tasa de recompra por puntuación RFM")
    plt.xlabel("Puntuación")
    plt.ylabel("Clientes que volvieron a comprar (%)")
    plt.xticks(range(1, 6))
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / "recompra_rfm.png",
        dpi=300
    )
    plt.close()

# GRÁFICA 3: Comparación de AUC.
def graficar_auc():
    plt.figure(figsize=(9, 5))

    barras = plt.bar(
        auc["Clasificador"],
        auc["AUC"]
    )

    plt.axhline(
        y=0.5,
        linestyle="--",
        label="Referencia aleatoria (AUC = 0.5)"
    )

    for barra, valor in zip(barras, auc["AUC"]):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.015,
            f"{valor:.4f}",
            ha="center"
        )

    plt.title("Comparación de AUC para predicción de recompra")
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.xticks(rotation=10)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / "comparacion_auc.png",
        dpi=300
    )
    plt.close()

# GRÁFICA 4: Participación en ventas netas.
def graficar_participacion_ventas():
    ventas_totales = float(
        nuevos.loc[
            nuevos["Indicador"] == "Ventas netas totales",
            "Valor"
        ].iloc[0]
    )

    ventas_nuevos = float(
        nuevos.loc[
            nuevos["Indicador"] == "Ventas netas de clientes nuevos",
            "Valor"
        ].iloc[0]
    )

    # Valor obtenido en el ejercicio 6.4.
    ventas_champions = 1295784.14

    # Las categorías son excluyentes:
    # un Champion ya compraba antes de octubre.
    ventas_resto = ventas_totales - ventas_champions - ventas_nuevos

    categorias = [
        "Champions",
        "Clientes nuevos",
        "Otros clientes"
    ]

    valores = [
        ventas_champions,
        ventas_nuevos,
        ventas_resto
    ]

    plt.figure(figsize=(8, 5))

    plt.pie(
        valores,
        labels=categorias,
        autopct="%1.2f%%",
        startangle=90
    )

    plt.title("Participación en las ventas netas de octubre a diciembre")
    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / "participacion_ventas.png",
        dpi=300
    )
    plt.close()


if __name__ == "__main__":
    graficar_quintiles()
    graficar_recompra()
    graficar_auc()
    graficar_participacion_ventas()

    print("\nGráficas generadas correctamente:")
    print("1. quintiles_rfm.png")
    print("2. recompra_rfm.png")
    print("3. comparacion_auc.png")
    print("4. participacion_ventas.png")
    print(f"\nCarpeta: {CARPETA_GRAFICAS}")