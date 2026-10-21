import pandas as pd
import importlib.util
from pathlib import Path

# Ubicación de los archivos y carpeta de resultados.
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO_LIMPIEZA = CARPETA_PROYECTO / "Parte 1" / "limpieza.py"
ARCHIVO_DATOS = CARPETA_PROYECTO / "Online Retail.xlsx"
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "Resultados"

# Importa las funciones de limpieza desde la Parte 1.
especificacion = importlib.util.spec_from_file_location("limpieza", ARCHIVO_LIMPIEZA)
modulo_limpieza = importlib.util.module_from_spec(especificacion)
especificacion.loader.exec_module(modulo_limpieza)

cargar_datos = modulo_limpieza.cargar_datos
limpiar = modulo_limpieza.limpiar

# Construye la serie de venta neta diaria.
def construir_serie_diaria(datos):
    datos_limpios = limpiar(datos).copy()

    # Calcula el importe de cada línea, incluyendo cancelaciones.
    datos_limpios["Importe"] = (
        datos_limpios["Quantity"] * datos_limpios["UnitPrice"]
    )
    datos_limpios["Fecha"] = datos_limpios["InvoiceDate"].dt.normalize()

    # Agrupa las ventas por día con al menos una transacción.
    ventas_diarias = (
        datos_limpios.groupby("Fecha")["Importe"]
        .agg(Venta_neta="sum", Transacciones="size")
        .sort_index()
    )

    # Construye el calendario completo entre la primera y última fecha.
    calendario = pd.date_range(
        start=ventas_diarias.index.min(),
        end=ventas_diarias.index.max(),
        freq="D"
    )

    serie_completa = ventas_diarias.reindex(calendario)
    serie_completa.index.name = "Fecha"

    return ventas_diarias, serie_completa

# Presenta los resultados del ejercicio 2.1.
def analizar_calendario(datos):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    ventas_diarias, serie_completa = construir_serie_diaria(datos)

    dias_calendario = len(serie_completa)
    dias_con_transacciones = len(ventas_diarias)
    dias_sin_transacciones = dias_calendario - dias_con_transacciones

    print("\nEjercicio 2.1: Serie de venta neta diaria")
    print(f"Primera fecha: {serie_completa.index.min().date()}")
    print(f"Última fecha: {serie_completa.index.max().date()}")
    print(f"Días calendario: {dias_calendario:,}")
    print(f"Días con transacciones: {dias_con_transacciones:,}")
    print(f"Días sin transacciones: {dias_sin_transacciones:,}")

    print("\nPrimeros diez días del calendario:")
    print(serie_completa.head(10).to_string())

    print("\nÚltimos diez días del calendario:")
    print(serie_completa.tail(10).to_string())

    # Conserva los días sin transacciones como datos faltantes.
    serie_completa.to_csv(
        CARPETA_RESULTADOS / "serie_venta_neta_diaria.csv",
        encoding="utf-8-sig"
    )

    return serie_completa

# Identifica los días sin transacciones y los compara con los feriados británicos.
def investigar_dias_sin_transacciones(serie_completa):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Registra los feriados de Inglaterra y Gales dentro del período.
    feriados = {
        "2010-12-27": "Navidad (día sustituto)",
        "2010-12-28": "Boxing Day (día sustituto)",
        "2011-01-03": "Año Nuevo (día sustituto)",
        "2011-04-22": "Viernes Santo",
        "2011-04-25": "Lunes de Pascua",
        "2011-04-29": "Boda real de William y Catherine",
        "2011-05-02": "Feriado de principios de mayo",
        "2011-05-30": "Feriado de primavera",
        "2011-08-29": "Feriado de verano"
    }

    # Extrae las fechas sin transacciones.
    dias_sin_ventas = serie_completa.loc[
        serie_completa["Transacciones"].isna()
    ].copy()

    dias_sin_ventas["Fecha"] = dias_sin_ventas.index
    dias_sin_ventas["Dia_semana"] = dias_sin_ventas.index.day_name()

    # Identifica los feriados conocidos.
    dias_sin_ventas["Feriado"] = (
        dias_sin_ventas.index.strftime("%Y-%m-%d")
        .map(feriados)
    )

    # Clasifica los días sin registros.
    def clasificar_dia(fila):
        if fila["Dia_semana"] == "Saturday":
            return "Cierre semanal"
        if pd.notna(fila["Feriado"]):
            return "Feriado bancario"
        return "Otro día sin transacciones"

    dias_sin_ventas["Clasificacion"] = dias_sin_ventas.apply(
        clasificar_dia, axis=1
    )

    print("\nEjercicio 2.2: Días sin transacciones")
    print(f"Total de días sin transacciones: {len(dias_sin_ventas)}")

    print("\nDistribución por día de la semana:")
    print(dias_sin_ventas["Dia_semana"].value_counts().to_string())

    print("\nClasificación de los días:")
    print(dias_sin_ventas["Clasificacion"].value_counts().to_string())

    print("\nLista completa de fechas sin transacciones:")
    print(
        dias_sin_ventas[
            ["Dia_semana", "Feriado", "Clasificacion"]
        ].fillna("-").to_string()
    )

    # Revisa también los feriados que sí registraron actividad.
    fechas_feriados = pd.to_datetime(list(feriados.keys()))
    fechas_feriados = fechas_feriados.intersection(
        serie_completa.index
    )

    revision_feriados = serie_completa.loc[
        fechas_feriados, ["Venta_neta", "Transacciones"]
    ].copy()

    revision_feriados["Feriado"] = (
        revision_feriados.index.strftime("%Y-%m-%d")
        .map(feriados)
    )

    revision_feriados["Tiene_transacciones"] = (
        revision_feriados["Transacciones"].notna()
    )

    print("\nRevisión de todos los feriados:")
    print(revision_feriados.to_string())

    # Guarda los resultados para documentarlos.
    dias_sin_ventas.to_csv(
        CARPETA_RESULTADOS / "dias_sin_transacciones.csv",
        index=False,
        encoding="utf-8-sig"
    )

    revision_feriados.to_csv(
        CARPETA_RESULTADOS / "revision_feriados.csv",
        encoding="utf-8-sig"
    )

    return dias_sin_ventas, revision_feriados

# Investiga si el último día del conjunto de datos está incompleto.
def investigar_ultimo_dia(datos, serie_completa):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    datos_limpios = limpiar(datos).copy()
    datos_limpios["Fecha"] = datos_limpios["InvoiceDate"].dt.normalize()

    ultima_fecha = datos_limpios["Fecha"].max()

    # Examina las transacciones del último día.
    ultimo_dia = datos_limpios.loc[
        datos_limpios["Fecha"] == ultima_fecha
    ].copy()

    # Compara las horas de inicio y fin de los últimos días con actividad.
    ultimos_dias = (
        datos_limpios.groupby("Fecha")
        .agg(
            Primera_transaccion=("InvoiceDate", "min"),
            Ultima_transaccion=("InvoiceDate", "max"),
            Lineas=("InvoiceNo", "size"),
            Facturas=("InvoiceNo", "nunique")
        )
        .tail(10)
    )

    print("\nEjercicio 2.4: Revisión del último día")
    print(f"Última fecha: {ultima_fecha.date()}")
    print(f"Primera transacción: {ultimo_dia['InvoiceDate'].min()}")
    print(f"Última transacción: {ultimo_dia['InvoiceDate'].max()}")
    print(f"Líneas registradas: {len(ultimo_dia):,}")
    print(f"Facturas diferentes: {ultimo_dia['InvoiceNo'].nunique():,}")
    print(f"Venta neta: £{serie_completa.loc[ultima_fecha, 'Venta_neta']:,.2f}")

    print("\nComparación de los últimos diez días con actividad:")
    print(ultimos_dias.to_string())

    # Prepara la serie para los ejercicios posteriores.
    serie_definitiva = serie_completa.loc[
        serie_completa.index < ultima_fecha
    ].copy()

    serie_definitiva.to_csv(
        CARPETA_RESULTADOS / "serie_diaria_sin_ultimo_dia.csv",
        encoding="utf-8-sig"
    )

    print("\nSerie preparada para las siguientes partes:")
    print(f"Primera fecha: {serie_definitiva.index.min().date()}")
    print(f"Última fecha: {serie_definitiva.index.max().date()}")
    print(f"Días calendario: {len(serie_definitiva):,}")
    print(f"Días sin transacciones: {serie_definitiva['Transacciones'].isna().sum():,}")

    return serie_definitiva

def main():
    datos = cargar_datos(ARCHIVO_DATOS)
    serie_completa = analizar_calendario(datos)
    investigar_dias_sin_transacciones(serie_completa)
    investigar_ultimo_dia(datos, serie_completa)

if __name__ == "__main__":
    main()