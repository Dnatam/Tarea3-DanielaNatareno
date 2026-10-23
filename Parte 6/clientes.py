import pandas as pd
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"

# Permite importar las funciones de limpieza de la Parte 1.
sys.path.insert(
    0,
    str(CARPETA_PROYECTO / "Parte 1")
)

from limpieza import cargar_datos, limpiar

ARCHIVO_DATOS = CARPETA_PROYECTO / "Online Retail.xlsx"
FECHA_CORTE = pd.Timestamp("2011-10-01")

def preparar_transacciones():
    # Carga los datos originales y aplica la limpieza de la Parte 1.
    datos = cargar_datos(ARCHIVO_DATOS)
    datos = limpiar(datos)

    # Conserva únicamente las transacciones anteriores al corte.
    datos = datos.loc[
        datos["InvoiceDate"] < FECHA_CORTE
    ].copy()

    # RFM requiere identificar a qué cliente pertenece cada operación.
    datos = datos.dropna(
        subset=["CustomerID"]
    ).copy()

    # Calcula el importe neto de cada línea.
    datos["Importe"] = (
        datos["Quantity"] * datos["UnitPrice"]
    )

    # Identifica las facturas de venta y las cancelaciones.
    datos["Es_venta"] = (
        datos["InvoiceNo"].str.match(r"^\d", na=False)
        & (datos["Quantity"] > 0)
    )

    return datos

def calcular_rfm():
    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    datos = preparar_transacciones()

    # Las compras determinan la recencia y la frecuencia.
    compras = datos.loc[
        datos["Es_venta"]
    ].copy()

    # Obtiene la última compra y el número de facturas distintas.
    resumen_compras = compras.groupby(
        "CustomerID"
    ).agg(
        Ultima_compra=("InvoiceDate", "max"),
        Frecuencia=("InvoiceNo", "nunique")
    )

    # El monto incluye ventas y cancelaciones conservadas.
    montos = datos.groupby(
        "CustomerID"
    )["Importe"].sum()

    # Redondea los importes a dos decimales para evitar errores de precisión.
    montos = montos.round(2)

    # Conserva a los clientes con al menos una factura de venta.
    rfm = resumen_compras.join(
        montos.rename("Monto")
    )

    # Calcula los días desde la última compra hasta la fecha de corte.
    rfm["Recencia"] = (
        FECHA_CORTE.normalize()
        - rfm["Ultima_compra"].dt.normalize()
    ).dt.days

    # Organiza las columnas del resultado.
    rfm = rfm[
        ["Recencia", "Frecuencia", "Monto", "Ultima_compra"]
    ].reset_index()

    rfm = rfm.sort_values(
        "CustomerID"
    ).reset_index(drop=True)

    # Identifica clientes cuyo monto neto no es positivo.
    monto_no_positivo = rfm["Monto"] <= 0

    print("\nEjercicio 6.1: Análisis RFM")
    print(f"Fecha de corte: {FECHA_CORTE.date()}")
    print(f"Transacciones limpias con cliente: {len(datos):,}")
    print(f"Clientes con al menos una compra: {len(rfm):,}")
    print(f"Clientes con M <= 0: {monto_no_positivo.sum():,}")

    print("\nPrimeros diez clientes:")
    print(rfm.head(10).to_string(index=False))

    print("\nEstadísticas de RFM:")
    print(
        rfm[["Recencia", "Frecuencia", "Monto"]]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nClientes con monto neto no positivo:")
    if monto_no_positivo.any():
        print(
            rfm.loc[monto_no_positivo]
            .to_string(index=False)
        )
    else:
        print("No se encontraron clientes con M <= 0.")

    # Guarda los resultados para utilizarlos en el ejercicio 6.2.
    rfm.to_csv(
        CARPETA_RESULTADOS / "clientes_rfm.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArchivo guardado: Resultados/clientes_rfm.csv")

    return rfm

def asignar_quintiles(rfm):
    # Ordena los clientes antes del desempate aleatorio.
    rfm = rfm.sort_values("CustomerID").reset_index(drop=True)

    # Asigna quintiles de recencia: menos días significa mejor puntuación.
    rfm["R_score"] = pd.qcut(
        rfm["Recencia"].rank(method="first", ascending=False),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Asigna quintiles de monto: mayor gasto significa mejor puntuación.
    rfm["M_score"] = pd.qcut(
        rfm["Monto"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Desempata la frecuencia mediante una permutación reproducible.
    generador = np.random.default_rng(2026)
    indices = generador.permutation(len(rfm))
    rfm_aleatorio = rfm.iloc[indices].copy()

    rfm_aleatorio["F_score"] = pd.qcut(
        rfm_aleatorio["Frecuencia"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Devuelve los clientes a su orden original.
    rfm = rfm_aleatorio.sort_values("CustomerID").reset_index(drop=True)

    # Calcula la suma de puntuaciones para utilizarla en el ejercicio 6.3.
    rfm["Puntaje_total"] = (
        rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    )

    print("\nEjercicio 6.2: Quintiles RFM")

    print("\nDistribución de puntuaciones:")
    for columna in ["R_score", "F_score", "M_score"]:
        print(f"\n{columna}:")
        print(rfm[columna].value_counts().sort_index().to_string())

    print("\nPrimeros diez clientes:")
    print(
        rfm[
            ["CustomerID", "Recencia", "Frecuencia", "Monto",
             "R_score", "F_score", "M_score", "Puntaje_total"]
        ].head(10).to_string(index=False)
    )

    # Muestra cómo se distribuyen los empates de frecuencia.
    print("\nClientes con una sola factura, por puntuación:")
    print(
        rfm.loc[rfm["Frecuencia"] == 1, "F_score"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Guarda los resultados para los siguientes ejercicios.
    rfm.to_csv(
        CARPETA_RESULTADOS / "clientes_rfm_quintiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArchivo guardado: Resultados/clientes_rfm_quintiles.csv")

    return rfm

def evaluar_recompra(rfm):
    # Carga los datos originales y aplica la misma limpieza de la Parte 1.
    datos = cargar_datos(ARCHIVO_DATOS)
    datos = limpiar(datos)

    # Selecciona el período de evaluación.
    fecha_final = pd.Timestamp("2011-12-09")

    datos = datos.loc[
        (datos["InvoiceDate"] >= FECHA_CORTE)
        & (datos["InvoiceDate"] < fecha_final)
        & datos["CustomerID"].notna()
    ].copy()

    # Solo las facturas de venta cuentan como una compra.
    ventas = datos.loc[
        datos["InvoiceNo"].str.match(r"^\d", na=False)
        & (datos["Quantity"] > 0)
    ].copy()

    # Identifica a los clientes que compraron durante la prueba.
    clientes_compra = set(ventas["CustomerID"].unique())

    rfm = rfm.copy()
    rfm["compra"] = rfm["CustomerID"].isin(clientes_compra).astype(int)

    print("\nEjercicio 6.3: Evaluación de recompra")
    print(f"Clientes evaluados: {len(rfm):,}")
    print(f"Clientes que volvieron a comprar: {rfm['compra'].sum():,}")
    print(f"Tasa de recompra: {rfm['compra'].mean():.2%}")

    # Calcula la fracción de recompra para cada quintil.
    for columna in ["R_score", "F_score", "M_score"]:
        resumen = rfm.groupby(columna)["compra"].agg(
            Clientes="count",
            Compraron="sum",
            Fraccion_recompra="mean"
        )

        print(f"\nRecompra según {columna}:")
        print(resumen.round(4).to_string())

        resumen.to_csv(
            CARPETA_RESULTADOS / f"recompra_{columna}.csv",
            encoding="utf-8-sig"
        )

    # Calcula el AUC usando los valores originales.
    resultados_auc = {
        "Recencia": roc_auc_score(rfm["compra"], -rfm["Recencia"]),
        "Frecuencia": roc_auc_score(rfm["compra"], rfm["Frecuencia"]),
        "Monto": roc_auc_score(rfm["compra"], rfm["Monto"]),
        "Puntaje RFM": roc_auc_score(rfm["compra"], rfm["Puntaje_total"])
    }

    print("\nAUC de los clasificadores:")
    for nombre, valor in resultados_auc.items():
        print(f"{nombre}: {valor:.4f}")

    # Guarda los resultados para los siguientes ejercicios.
    rfm.to_csv(
        CARPETA_RESULTADOS / "clientes_rfm_recompra.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        list(resultados_auc.items()),
        columns=["Clasificador", "AUC"]
    ).to_csv(
        CARPETA_RESULTADOS / "resultados_auc.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return rfm

def analizar_champions(rfm):
    # Identifica clientes con puntuaciones 4 o 5 en R, F y M.
    rfm = rfm.copy()

    rfm["Champion"] = (
        (rfm["R_score"] >= 4)
        & (rfm["F_score"] >= 4)
        & (rfm["M_score"] >= 4)
    )

    clientes_champions = set(
        rfm.loc[rfm["Champion"], "CustomerID"]
    )

    cantidad = len(clientes_champions)
    porcentaje_clientes = cantidad / len(rfm) * 100

    # Carga los datos y aplica la limpieza original.
    datos = cargar_datos(ARCHIVO_DATOS)
    datos = limpiar(datos)

    # Selecciona las transacciones del período de evaluación.
    fecha_final = pd.Timestamp("2011-12-09")

    datos = datos.loc[
        (datos["InvoiceDate"] >= FECHA_CORTE)
        & (datos["InvoiceDate"] < fecha_final)
        & datos["CustomerID"].notna()
    ].copy()

    # Incluye ventas y cancelaciones para calcular ventas netas.
    datos["Importe"] = datos["Quantity"] * datos["UnitPrice"]

    ventas_totales = datos["Importe"].sum()

    ventas_champions = datos.loc[
        datos["CustomerID"].isin(clientes_champions),
        "Importe"
    ].sum()

    porcentaje_ventas = ventas_champions / ventas_totales * 100

    print("\nEjercicio 6.4: Clientes Champions")
    print(f"Clientes Champions: {cantidad:,}")
    print(f"Porcentaje de clientes Champions: {porcentaje_clientes:.2f}%")
    print(f"Ventas netas totales: {ventas_totales:,.2f}")
    print(f"Ventas netas de Champions: {ventas_champions:,.2f}")
    print(f"Porcentaje de ventas de Champions: {porcentaje_ventas:.2f}%")

    # Guarda la identificación de los clientes Champions.
    rfm.to_csv(
        CARPETA_RESULTADOS / "clientes_champions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return rfm

def analizar_clientes_nuevos():
    # Carga los datos y aplica la limpieza de la Parte 1.
    datos = cargar_datos(ARCHIVO_DATOS)
    datos = limpiar(datos)

    # Conserva las transacciones con clientes identificados.
    datos = datos.dropna(subset=["CustomerID"]).copy()

    # Calcula el importe neto de cada transacción.
    datos["Importe"] = datos["Quantity"] * datos["UnitPrice"]

    # Identifica las facturas de venta.
    datos["Es_venta"] = (
        datos["InvoiceNo"].str.match(r"^\d", na=False)
        & (datos["Quantity"] > 0)
    )

    # Identifica clientes que ya habían comprado antes del corte.
    compras_anteriores = datos.loc[
        (datos["InvoiceDate"] < FECHA_CORTE)
        & datos["Es_venta"]
    ]

    clientes_antiguos = set(compras_anteriores["CustomerID"])

    # Selecciona el período de evaluación.
    fecha_final = pd.Timestamp("2011-12-09")

    periodo = datos.loc[
        (datos["InvoiceDate"] >= FECHA_CORTE)
        & (datos["InvoiceDate"] < fecha_final)
    ].copy()

    # Identifica a quienes realizaron su primera compra en el período.
    compras_periodo = periodo.loc[periodo["Es_venta"]]

    clientes_nuevos = set(compras_periodo["CustomerID"]) - clientes_antiguos

    # Calcula ventas netas incluyendo devoluciones.
    ventas_totales = periodo["Importe"].sum()

    ventas_nuevos = periodo.loc[
        periodo["CustomerID"].isin(clientes_nuevos),
        "Importe"
    ].sum()

    porcentaje_nuevos = ventas_nuevos / ventas_totales * 100

    print("\nEjercicio 6.5: Clientes nuevos")
    print(f"Clientes nuevos: {len(clientes_nuevos):,}")
    print(f"Ventas netas totales: {ventas_totales:,.2f}")
    print(f"Ventas netas de clientes nuevos: {ventas_nuevos:,.2f}")
    print(f"Porcentaje de ventas de clientes nuevos: {porcentaje_nuevos:.2f}%")

    # Guarda el resumen.
    resumen = pd.DataFrame({
        "Indicador": [
            "Clientes nuevos",
            "Ventas netas totales",
            "Ventas netas de clientes nuevos",
            "Porcentaje de ventas de clientes nuevos"
        ],
        "Valor": [
            len(clientes_nuevos),
            round(ventas_totales, 2),
            round(ventas_nuevos, 2),
            round(porcentaje_nuevos, 2)
        ]
    })

    resumen.to_csv(
        CARPETA_RESULTADOS / "clientes_nuevos_resumen.csv",
        index=False,
        encoding="utf-8-sig"
    )

if __name__ == "__main__":
    rfm = calcular_rfm()
    rfm = asignar_quintiles(rfm)
    rfm = evaluar_recompra(rfm)
    analizar_champions(rfm)
    analizar_clientes_nuevos()