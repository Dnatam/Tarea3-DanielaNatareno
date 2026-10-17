
import pandas as pd
from pathlib import Path

# Ubicación del archivo original y carpeta de resultados.
ARCHIVO_DATOS = Path(__file__).resolve().parent / "Online Retail.xlsx"
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"

# Carga las transacciones conservando los identificadores como texto.
def cargar_datos(ruta_archivo):
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
    tipos_columnas = {
        "InvoiceNo": "string",
        "StockCode": "string",
        "CustomerID": "string"
    }
    datos = pd.read_excel(ruta_archivo, dtype=tipos_columnas, engine="openpyxl")
    datos["InvoiceDate"] = pd.to_datetime(datos["InvoiceDate"], errors="coerce")
    return datos

# Calcula las cantidades solicitadas antes de aplicar la limpieza.
def explorar_datos(datos):
    total_filas = len(datos)
    total_facturas = datos["InvoiceNo"].nunique(dropna=True)
    total_productos = datos["StockCode"].nunique(dropna=True)
    total_clientes = datos["CustomerID"].nunique(dropna=True)
    clientes_faltantes = datos["CustomerID"].isna().sum()
    fraccion_faltante = clientes_faltantes / total_filas

    print("Exploración inicial de Online Retail")
    print(f"Total de filas: {total_filas:,}")
    print(f"Facturas diferentes: {total_facturas:,}")
    print(f"Productos diferentes: {total_productos:,}")
    print(f"Clientes diferentes: {total_clientes:,}")
    print(f"Filas sin CustomerID: {clientes_faltantes:,}")
    print(f"Fracción sin CustomerID: {fraccion_faltante:.6f}")
    print(f"Porcentaje sin CustomerID: {fraccion_faltante:.2%}")
    print("\nTipos de datos:")
    print(datos.dtypes.to_string())
    print("\nPrimeras cinco filas:")
    print(datos.head().to_string(index=False))

# Muestra ejemplos de registros para investigar su significado.
def mostrar_ejemplos(datos, columnas, cantidad=10):
    if datos.empty:
        print("No se encontraron registros para esta condición.")
        return
    print(datos[columnas].head(cantidad).to_string(index=False))

# Investiga las anomalías solicitadas sin modificar los datos originales.
def investigar_anomalias(datos):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    columnas = ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID"]
    print("\nInvestigación de anomalías del ejercicio 1.2")

    # Identifica filas duplicadas exactas considerando todas las columnas.
    duplicados = datos.duplicated(keep="first")
    print("\na) Filas duplicadas exactas")
    print(f"Filas duplicadas adicionales: {duplicados.sum():,}")
    mostrar_ejemplos(datos.loc[duplicados], columnas)

    # Identifica facturas que no comienzan con un dígito y examina sus prefijos.
    facturas_especiales = ~datos["InvoiceNo"].str.match(r"^\d", na=False)
    print("\nb) Facturas que no comienzan con un dígito")
    print(f"Filas encontradas: {facturas_especiales.sum():,}")
    print(f"Facturas diferentes: {datos.loc[facturas_especiales, 'InvoiceNo'].nunique():,}")

    prefijos = datos.loc[facturas_especiales, "InvoiceNo"].str.extract(
        r"^([^\d]+)", expand=False
    )
    print("\nPrefijos identificados:")
    print(prefijos.value_counts(dropna=False).to_string())
    print("\nEjemplos por prefijo:")

    for prefijo in prefijos.dropna().unique():
        seleccion = facturas_especiales & datos["InvoiceNo"].str.startswith(
            prefijo, na=False
        )
        print(f"\nPrefijo: {prefijo}")
        mostrar_ejemplos(datos.loc[seleccion], columnas, cantidad=5)

    # Examina códigos que no comienzan con cinco dígitos.
    codigos_especiales = ~datos["StockCode"].str.match(r"^\d{5}", na=False)
    print("\nc) Códigos StockCode especiales")
    print(f"Filas encontradas: {codigos_especiales.sum():,}")
    print(f"Códigos diferentes: {datos.loc[codigos_especiales, 'StockCode'].nunique():,}")

    resumen_codigos = datos.loc[codigos_especiales].groupby(
        ["StockCode", "Description"], dropna=False
    ).agg(
        filas=("StockCode", "size"),
        cantidad_total=("Quantity", "sum")
    ).reset_index().sort_values("filas", ascending=False)

    print("\nCódigos y descripciones más frecuentes:")
    print(resumen_codigos.head(40).to_string(index=False))
    resumen_codigos.to_csv(
        CARPETA_RESULTADOS / "codigos_especiales.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Distingue los registros con precios cero de aquellos con precios negativos.
    precios_cero = datos["UnitPrice"] == 0
    precios_negativos = datos["UnitPrice"] < 0
    print("\nd) Precios iguales a cero o negativos")
    print(f"Filas con precio cero: {precios_cero.sum():,}")
    print(f"Filas con precio negativo: {precios_negativos.sum():,}")
    print("\nEjemplos con precio cero:")
    mostrar_ejemplos(datos.loc[precios_cero], columnas)
    print("\nEjemplos con precio negativo:")
    mostrar_ejemplos(datos.loc[precios_negativos], columnas)

    # Investiga cantidades negativas que no pertenecen a facturas de cancelación.
    cantidades_negativas = datos["Quantity"] < 0
    cancelaciones = datos["InvoiceNo"].str.startswith("C", na=False)
    negativos_sin_cancelacion = cantidades_negativas & ~cancelaciones

    print("\ne) Cantidades negativas sin factura de cancelación")
    print(f"Filas encontradas: {negativos_sin_cancelacion.sum():,}")
    print("\nDescripciones más frecuentes:")
    print(
        datos.loc[negativos_sin_cancelacion, "Description"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )
    print("\nEjemplos:")
    mostrar_ejemplos(datos.loc[negativos_sin_cancelacion], columnas, cantidad=15)

    # Examina los registros sin identificador de cliente.
    clientes_faltantes = datos["CustomerID"].isna()
    print("\nf) Filas sin CustomerID")
    print(f"Filas encontradas: {clientes_faltantes.sum():,}")
    print(f"Porcentaje del total: {clientes_faltantes.mean():.2%}")
    print(f"Filas sin cliente y con cancelación: {(clientes_faltantes & cancelaciones).sum():,}")
    print(f"Filas sin cliente y sin cancelación: {(clientes_faltantes & ~cancelaciones).sum():,}")
    print("\nEjemplos:")
    mostrar_ejemplos(datos.loc[clientes_faltantes], columnas)

# Identifica movimientos administrativos que no representan ventas comerciales.
def identificar_movimientos_administrativos(datos):
    codigos_administrativos = {
        "BANK CHARGES",
        "AMAZONFEE",
        "CRUK",
        "S"
    }
    codigos = datos["StockCode"].str.strip().str.upper()
    descripciones = datos["Description"].fillna("").str.strip().str.lower()
    descripciones_administrativas = {
        "adjust bad debt",
        "bank charges",
        "amazon fee",
        "cruk commission",
        "samples"
    }
    return (
        codigos.isin(codigos_administrativos)
        | descripciones.isin(descripciones_administrativas)
    ).fillna(False)

# Aplica una regla y registra su efecto sobre las filas y el importe.
def aplicar_regla(datos, condicion, nombre_regla, bitacora):
    condicion = condicion.fillna(False)
    filas_eliminadas = datos.loc[condicion]
    monto_eliminado = (
        filas_eliminadas["Quantity"] * filas_eliminadas["UnitPrice"]
    ).sum()

    bitacora.append({
        "Orden": len(bitacora) + 1,
        "Regla": nombre_regla,
        "Filas eliminadas": len(filas_eliminadas),
        "Monto eliminado (£)": round(float(monto_eliminado), 2),
        "Filas restantes": len(datos) - len(filas_eliminadas)
    })
    return datos.loc[~condicion].copy()

# Limpia las transacciones y genera opcionalmente una bitácora de auditoría.
def limpiar(datos, devolver_bitacora=False):
    datos_limpios = datos.copy()
    bitacora = []

    # Regla 1: eliminar filas duplicadas exactas.
    condicion = datos_limpios.duplicated(keep="first")
    datos_limpios = aplicar_regla(
        datos_limpios, condicion, "Duplicados exactos", bitacora
    )

    # Regla 2: excluir facturas correspondientes a ajustes de deuda.
    condicion = datos_limpios["InvoiceNo"].str.startswith("A", na=False)
    datos_limpios = aplicar_regla(
        datos_limpios, condicion, "Facturas con prefijo A", bitacora
    )

    # Regla 3: excluir movimientos administrativos identificados.
    condicion = identificar_movimientos_administrativos(datos_limpios)
    datos_limpios = aplicar_regla(
        datos_limpios, condicion, "Movimientos administrativos", bitacora
    )

    # Regla 4: eliminar registros sin un precio comercial positivo.
    condicion = (
        datos_limpios["UnitPrice"].isna()
        | (datos_limpios["UnitPrice"] <= 0)
    )
    datos_limpios = aplicar_regla(
        datos_limpios, condicion, "Precio cero, negativo o faltante", bitacora
    )

    # Regla 5: eliminar cantidades negativas que no son cancelaciones.
    cancelaciones = datos_limpios["InvoiceNo"].str.startswith("C", na=False)
    condicion = (datos_limpios["Quantity"] < 0) & ~cancelaciones
    datos_limpios = aplicar_regla(
        datos_limpios, condicion, "Cantidad negativa sin cancelación", bitacora
    )

    # Regla 6: excluir registros sin información esencial para calcular ventas.
    condicion = (
        datos_limpios["InvoiceNo"].isna()
        | datos_limpios["StockCode"].isna()
        | datos_limpios["InvoiceDate"].isna()
        | datos_limpios["Quantity"].isna()
    )
    datos_limpios = aplicar_regla(
        datos_limpios, condicion, "Datos esenciales faltantes", bitacora
    )

    # Conserva CustomerID faltantes porque no invalidan las ventas.
    if devolver_bitacora:
        return datos_limpios, pd.DataFrame(bitacora)
    return datos_limpios

# Genera y guarda el resumen de las reglas aplicadas.
def generar_bitacora(datos):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    datos_limpios, bitacora = limpiar(datos, devolver_bitacora=True)
    monto_original = (datos["Quantity"] * datos["UnitPrice"]).sum()
    monto_limpio = (
        datos_limpios["Quantity"] * datos_limpios["UnitPrice"]
    ).sum()

    print("\nBitácora de limpieza del ejercicio 1.3")
    print(bitacora.to_string(index=False))
    print("\nResumen general:")
    print(f"Filas originales: {len(datos):,}")
    print(f"Filas eliminadas: {len(datos) - len(datos_limpios):,}")
    print(f"Filas conservadas: {len(datos_limpios):,}")
    print(f"Monto original: £{monto_original:,.2f}")
    print(f"Monto eliminado: £{monto_original - monto_limpio:,.2f}")
    print(f"Monto conservado: £{monto_limpio:,.2f}")
    print(f"Filas sin CustomerID conservadas: {datos_limpios['CustomerID'].isna().sum():,}")

    # Guarda la bitácora y el conjunto limpio para los siguientes ejercicios.
    bitacora.to_csv(
        CARPETA_RESULTADOS / "bitacora_limpieza.csv",
        index=False,
        encoding="utf-8-sig"
    )
    datos_limpios.to_csv(
        CARPETA_RESULTADOS / "online_retail_limpio.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return datos_limpios, bitacora

# Busca transacciones relacionadas por cliente o producto en un período determinado.
def buscar_transacciones_relacionadas(datos, transaccion, dias=30):
    fecha = transaccion["InvoiceDate"]
    codigo = transaccion["StockCode"]
    cliente = transaccion["CustomerID"]

    # Busca movimientos cercanos a la fecha de la transacción extrema.
    inicio = fecha - pd.Timedelta(days=dias)
    fin = fecha + pd.Timedelta(days=dias)
    periodo = datos["InvoiceDate"].between(inicio, fin)

    # Incluye el mismo producto y, cuando existe, las operaciones del mismo cliente.
    mismo_producto = datos["StockCode"].eq(codigo).fillna(False)
    if pd.notna(cliente):
        mismo_cliente = datos["CustomerID"].eq(cliente).fillna(False)
        condicion = periodo & (mismo_producto | mismo_cliente)
    else:
        condicion = periodo & mismo_producto

    relacionadas = datos.loc[condicion].copy()
    relacionadas["Importe"] = relacionadas["Quantity"] * relacionadas["UnitPrice"]
    relacionadas["Diferencia_minutos"] = (
        relacionadas["InvoiceDate"] - fecha
    ).dt.total_seconds() / 60
    relacionadas["Mismo_producto"] = relacionadas["StockCode"].eq(codigo).fillna(False)
    relacionadas["Mismo_cliente"] = (
        relacionadas["CustomerID"].eq(cliente).fillna(False)
        if pd.notna(cliente)
        else False
    )
    relacionadas = relacionadas.sort_values("InvoiceDate")
    return relacionadas

# Investiga las seis líneas de mayor importe absoluto antes de aplicar la limpieza.
def investigar_transacciones_extremas(datos):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Calcula los importes y la fecha antes de seleccionar las seis líneas extremas.
    datos_originales = datos.copy()
    datos_originales["Importe"] = (
        datos_originales["Quantity"] * datos_originales["UnitPrice"]
    )
    datos_originales["Importe_absoluto"] = datos_originales["Importe"].abs()
    datos_originales["Fecha"] = datos_originales["InvoiceDate"].dt.normalize()
    extremos = datos_originales.nlargest(6, "Importe_absoluto").copy()

    # Comprueba si cada transacción extrema permanece después de la limpieza.
    datos_limpios = limpiar(datos)
    indices_conservados = datos_limpios.index
    extremos["Conservada_tras_limpieza"] = extremos.index.isin(indices_conservados)

    # Calcula las ventas netas diarias del conjunto original.
    ventas_diarias_originales = (
        datos_originales.groupby("Fecha")["Importe"].sum()
    )

    # Calcula las ventas netas diarias del conjunto limpio.
    datos_limpios = datos_limpios.copy()
    datos_limpios["Importe"] = (
        datos_limpios["Quantity"] * datos_limpios["UnitPrice"]
    )
    datos_limpios["Fecha"] = datos_limpios["InvoiceDate"].dt.normalize()
    ventas_diarias_limpias = (
        datos_limpios.groupby("Fecha")["Importe"].sum()
    )

    columnas_extremos = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "UnitPrice", "InvoiceDate", "CustomerID", "Importe",
        "Importe_absoluto", "Conservada_tras_limpieza"
    ]

    print("\nEjercicio 1.4: Seis transacciones de mayor importe absoluto")
    print(extremos[columnas_extremos].to_string(index=False))

    # Examina las facturas y operaciones relacionadas de cada línea extrema.
    resumen = []
    for posicion, (indice, transaccion) in enumerate(extremos.iterrows(), start=1):
        fecha = transaccion["Fecha"]
        importe = transaccion["Importe"]
        total_original = ventas_diarias_originales.loc[fecha]
        total_limpio = ventas_diarias_limpias.get(fecha, 0.0)
        otras_lineas = total_original - importe

        resumen.append({
            "Posicion": posicion,
            "Indice_original": indice,
            "InvoiceNo": transaccion["InvoiceNo"],
            "StockCode": transaccion["StockCode"],
            "Fecha": fecha.date(),
            "Importe_extremo": round(float(importe), 2),
            "Otras_lineas_del_dia": round(float(otras_lineas), 2),
            "Total_dia_original": round(float(total_original), 2),
            "Total_dia_limpio": round(float(total_limpio), 2),
            "Conservada_tras_limpieza": bool(transaccion["Conservada_tras_limpieza"])
        })

        print(f"\nTransacción extrema {posicion}")
        print(f"Factura: {transaccion['InvoiceNo']}")
        print(f"Producto: {transaccion['StockCode']}")
        print(f"Descripción: {transaccion['Description']}")
        print(f"Fecha: {transaccion['InvoiceDate']}")
        print(f"Cantidad: {transaccion['Quantity']:,}")
        print(f"Precio unitario: £{transaccion['UnitPrice']:,.2f}")
        print(f"Importe: £{importe:,.2f}")
        print(f"Conservada tras limpieza: {transaccion['Conservada_tras_limpieza']}")
        print(f"Total neto del día original: £{total_original:,.2f}")
        print(f"Total neto del día limpio: £{total_limpio:,.2f}")

        # Muestra las líneas de la misma factura.
        misma_factura = datos_originales.loc[
            datos_originales["InvoiceNo"].eq(transaccion["InvoiceNo"]).fillna(False)
        ].copy()
        print("\nLíneas de la misma factura:")
        print(
            misma_factura[
                ["InvoiceNo", "StockCode", "Description", "Quantity",
                 "UnitPrice", "InvoiceDate", "Importe"]
            ].head(20).to_string(index=False)
        )

        # Guarda las operaciones relacionadas para investigar compensaciones.
        relacionadas = buscar_transacciones_relacionadas(
            datos_originales, transaccion
        )
        relacionadas.to_csv(
            CARPETA_RESULTADOS / f"extremo_{posicion}_relacionadas.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nTransacciones relacionadas encontradas: {len(relacionadas):,}")
        columnas_relacionadas = [
            "InvoiceNo", "StockCode", "Description", "Quantity",
            "UnitPrice", "InvoiceDate", "CustomerID", "Importe",
            "Diferencia_minutos", "Mismo_producto", "Mismo_cliente"
        ]
        print(relacionadas[columnas_relacionadas].head(30).to_string(index=False))

        # Busca compensaciones exactas del importe dentro del mismo cliente y día.
        if pd.notna(transaccion["CustomerID"]):
            mismo_cliente = datos_originales["CustomerID"].eq(
                transaccion["CustomerID"]
            ).fillna(False)
            mismo_dia = datos_originales["Fecha"].eq(fecha)
            importe_opuesto = (
                datos_originales["Importe"] + importe
            ).abs() < 0.005
            compensaciones = datos_originales.loc[
                mismo_cliente & mismo_dia & importe_opuesto
            ].copy()
        else:
            compensaciones = datos_originales.iloc[0:0].copy()

        print("\nPosibles compensaciones exactas del mismo cliente y día:")
        if compensaciones.empty:
            print("No se encontraron compensaciones exactas.")
        else:
            print(
                compensaciones[
                    ["InvoiceNo", "StockCode", "Description", "Quantity",
                     "UnitPrice", "InvoiceDate", "CustomerID", "Importe"]
                ].to_string(index=False)
            )

    # Guarda las seis líneas originales y el resumen de su efecto diario.
    resumen = pd.DataFrame(resumen)
    extremos[columnas_extremos].to_csv(
        CARPETA_RESULTADOS / "seis_transacciones_extremas.csv",
        index=False,
        encoding="utf-8-sig"
    )
    resumen.to_csv(
        CARPETA_RESULTADOS / "resumen_transacciones_extremas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResumen de las seis transacciones extremas")
    print(resumen.to_string(index=False))
    return extremos, resumen

# Comprueba que la limpieza es reutilizable y produce resultados consistentes.
def verificar_limpieza(datos):
    datos_originales = datos.copy(deep=True)

    primera_limpieza = limpiar(datos)
    segunda_limpieza = limpiar(datos)

    mismo_resultado = primera_limpieza.equals(segunda_limpieza)
    original_sin_cambios = datos.equals(datos_originales)

    # Comprueba que repetir la limpieza no elimina filas adicionales.
    limpieza_repetida = limpiar(primera_limpieza)
    resultado_estable = primera_limpieza.equals(limpieza_repetida)

    print("\nVerificación de la función limpiar()")
    print(f"El DataFrame original permanece sin cambios: {original_sin_cambios}")
    print(f"Dos ejecuciones producen el mismo resultado: {mismo_resultado}")
    print(f"Repetir la limpieza mantiene el resultado: {resultado_estable}")
    print(f"Filas originales: {len(datos):,}")
    print(f"Filas limpias: {len(primera_limpieza):,}")

    return original_sin_cambios and mismo_resultado and resultado_estable

def main():
    datos = cargar_datos(ARCHIVO_DATOS)
    explorar_datos(datos)
    investigar_anomalias(datos)
    generar_bitacora(datos)
    investigar_transacciones_extremas(datos)
    verificar_limpieza(datos)

if __name__ == "__main__":
    main()
