import io
import re
import unicodedata
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Historial de Aspirantes",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Módulo 1 · Historial de Aspirantes")
st.caption(
    "Carga, integración y análisis básico de aspirantes por carrera."
)


# ============================================================
# FUNCIONES DE APOYO
# ============================================================

def limpiar_texto(valor):
    """Convierte un texto a formato comparable: minúsculas, sin acentos."""
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return re.sub(r"\s+", " ", texto)


def buscar_fila_encabezados(df_crudo):
    """
    Busca automáticamente la fila que contiene encabezados.
    La referencia principal es la columna Matrícula/ID.
    """
    palabras_clave = [
        "matricula/id",
        "matricula",
        "id",
        "apellido paterno",
        "nombre (s)"
    ]

    for indice, fila in df_crudo.iterrows():
        valores = [limpiar_texto(valor) for valor in fila.tolist()]
        coincidencias = sum(
            any(palabra in valor for valor in valores)
            for palabra in palabras_clave
        )

        if coincidencias >= 2:
            return indice

    return None


def obtener_nombre_carrera(nombre_hoja, df_crudo):
    """
    Intenta recuperar el nombre de carrera desde la hoja.
    Si no encuentra una celda con 'CARRERA', usa el nombre de la hoja.
    """
    for _, fila in df_crudo.iterrows():
        valores = fila.tolist()

        for i, valor in enumerate(valores):
            if limpiar_texto(valor) == "carrera":
                if i + 1 < len(valores) and pd.notna(valores[i + 1]):
                    return str(valores[i + 1]).strip()

    return str(nombre_hoja).strip()


def encontrar_columna(df, posibles_nombres):
    """Encuentra una columna aunque cambien mayúsculas, acentos o espacios."""
    columnas_limpias = {
        limpiar_texto(columna): columna
        for columna in df.columns
    }

    for posible in posibles_nombres:
        posible_limpio = limpiar_texto(posible)

        for columna_limpia, columna_original in columnas_limpias.items():
            if posible_limpio == columna_limpia:
                return columna_original

        for columna_limpia, columna_original in columnas_limpias.items():
            if posible_limpio in columna_limpia:
                return columna_original

    return None


def convertir_promedio(valor):
    """
    Normaliza el promedio a escala 0-100.

    Reglas:
    - 0 a 10: se multiplica por 10.
    - Mayor a 10 y hasta 100: se conserva.
    - Fuera de rango, fechas o texto no numérico: dato dudoso.
    """

    if pd.isna(valor) or str(valor).strip() == "":
        return np.nan, "Sin dato"

    if isinstance(valor, (datetime, date, pd.Timestamp)):
        return np.nan, "Dato dudoso: formato fecha"

    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")

    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return np.nan, "Dato dudoso: no numérico"

    if 0 <= numero <= 10:
        return round(numero * 10, 2), "Válido: convertido de escala 0-10"

    if 10 < numero <= 100:
        return round(numero, 2), "Válido: escala 0-100"

    return np.nan, "Dato dudoso: fuera de rango"


def procesar_hoja(archivo, nombre_hoja):
    """
    Lee una hoja sin perder columnas originales.
    Detecta encabezados, conserva todas las variables y añade Carrera.
    """

    df_crudo = pd.read_excel(
        archivo,
        sheet_name=nombre_hoja,
        header=None,
        dtype=object
    )

    fila_encabezados = buscar_fila_encabezados(df_crudo)

    if fila_encabezados is None:
        return None, {
            "hoja": nombre_hoja,
            "estatus": "No procesada",
            "detalle": "No se identificó la fila de encabezados."
        }

    carrera = obtener_nombre_carrera(nombre_hoja, df_crudo)

    encabezados = df_crudo.iloc[fila_encabezados].tolist()
    encabezados_limpios = []

    for posicion, encabezado in enumerate(encabezados):
        if pd.isna(encabezado) or str(encabezado).strip() == "":
            encabezados_limpios.append(f"Columna_sin_nombre_{posicion + 1}")
        else:
            encabezados_limpios.append(str(encabezado).strip())

    df = df_crudo.iloc[fila_encabezados + 1:].copy()
    df.columns = encabezados_limpios

    # No se eliminan columnas.
    # Solo se eliminan filas completamente vacías.
    df = df.dropna(how="all").copy()

    columna_id = encontrar_columna(
        df,
        ["Matrícula/ID", "Matrícula", "ID"]
    )

    # Conservamos únicamente filas de personas identificables.
    # Esto evita traer notas, totales o celdas administrativas al análisis.
    if columna_id is not None:
        df = df[df[columna_id].notna()].copy()

    # Se añade carrera sin borrar ningún encabezado existente.
    df["Carrera"] = carrera
    df["Hoja_origen"] = nombre_hoja

    columna_promedio = encontrar_columna(
        df,
        ["Promedio Bachillerato", "Promedio", "Promedio de Bachillerato"]
    )

    if columna_promedio is not None:
        resultado_promedio = df[columna_promedio].apply(convertir_promedio)

        df["Promedio_normalizado_100"] = resultado_promedio.apply(
            lambda x: x[0]
        )

        df["Estatus_promedio"] = resultado_promedio.apply(
            lambda x: x[1]
        )
    else:
        df["Promedio_normalizado_100"] = np.nan
        df["Estatus_promedio"] = "Sin columna de promedio"

    return df, {
        "hoja": nombre_hoja,
        "estatus": "Procesada",
        "detalle": f"{len(df)} registros identificados."
    }


@st.cache_data(show_spinner=False)
def procesar_archivo_excel(contenido_archivo):
    """
    Integra todas las hojas en un solo DataFrame.
    """

    archivo = io.BytesIO(contenido_archivo)
    excel = pd.ExcelFile(archivo)

    bases = []
    bitacora = []

    for hoja in excel.sheet_names:
        df_hoja, resultado = procesar_hoja(
            io.BytesIO(contenido_archivo),
            hoja
        )

        bitacora.append(resultado)

        if df_hoja is not None and not df_hoja.empty:
            bases.append(df_hoja)

    if not bases:
        return pd.DataFrame(), pd.DataFrame(bitacora)

    df_general = pd.concat(
        bases,
        ignore_index=True,
        sort=False
    )

    return df_general, pd.DataFrame(bitacora)


def convertir_excel_descargable(df):
    """Convierte un DataFrame a archivo Excel descargable."""
    salida = io.BytesIO()

    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Aspirantes_integrados"
        )

    return salida.getvalue()


# ============================================================
# MENÚ LATERAL
# ============================================================

st.sidebar.header("Menú")

seccion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Carga de archivo",
        "Panorama general",
        "Lista por carrera",
        "Calidad de datos",
        "Base integrada"
    ]
)


# ============================================================
# CARGA DEL ARCHIVO
# ============================================================

archivo_subido = st.sidebar.file_uploader(
    "Carga el archivo Excel de aspirantes",
    type=["xlsx", "xls"]
)

if archivo_subido is None:
    st.info(
        "Carga un archivo Excel desde el menú lateral para iniciar el análisis."
    )
    st.stop()

contenido_archivo = archivo_subido.getvalue()

with st.spinner("Leyendo e integrando hojas..."):
    df_general, df_bitacora = procesar_archivo_excel(contenido_archivo)

if df_general.empty:
    st.error(
        "No se pudieron identificar registros de aspirantes en el archivo."
    )
    st.dataframe(df_bitacora, use_container_width=True)
    st.stop()


# ============================================================
# SECCIÓN 1 · CARGA
# ============================================================

if seccion == "Carga de archivo":

    st.subheader("Archivo procesado")

    col1, col2, col3 = st.columns(3)

    col1.metric("Hojas detectadas", len(df_bitacora))
    col2.metric("Hojas procesadas", (df_bitacora["estatus"] == "Procesada").sum())
    col3.metric("Aspirantes integrados", len(df_general))

    st.markdown("#### Bitácora de lectura")
    st.dataframe(df_bitacora, use_container_width=True)

    st.success(
        "Las columnas originales fueron conservadas. "
        "Solo se añadieron: Carrera, Hoja_origen, "
        "Promedio_normalizado_100 y Estatus_promedio."
    )


# ============================================================
# SECCIÓN 2 · PANORAMA GENERAL
# ============================================================

elif seccion == "Panorama general":

    st.subheader("Estadística básica de aspirantes")

    total_aspirantes = len(df_general)
    total_carreras = df_general["Carrera"].nunique()

    promedio_general = df_general["Promedio_normalizado_100"].mean()
    mediana_general = df_general["Promedio_normalizado_100"].median()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Aspirantes", f"{total_aspirantes:,}")
    col2.metric("Carreras", total_carreras)
    col3.metric(
        "Promedio general",
        f"{promedio_general:.2f}" if pd.notna(promedio_general) else "Sin dato"
    )
    col4.metric(
        "Mediana",
        f"{mediana_general:.2f}" if pd.notna(mediana_general) else "Sin dato"
    )

    st.markdown("#### Aspirantes por carrera")

    resumen_carrera = (
        df_general
        .groupby("Carrera", dropna=False)
        .agg(
            Aspirantes=("Carrera", "size"),
            Promedio=("Promedio_normalizado_100", "mean"),
            Mediana=("Promedio_normalizado_100", "median"),
            Mínimo=("Promedio_normalizado_100", "min"),
            Máximo=("Promedio_normalizado_100", "max")
        )
        .reset_index()
        .sort_values("Aspirantes", ascending=False)
    )

    for columna in ["Promedio", "Mediana", "Mínimo", "Máximo"]:
        resumen_carrera[columna] = resumen_carrera[columna].round(2)

    st.dataframe(resumen_carrera, use_container_width=True)

    st.bar_chart(
        resumen_carrera.set_index("Carrera")["Aspirantes"]
    )


# ============================================================
# SECCIÓN 3 · LISTA POR CARRERA
# ============================================================

elif seccion == "Lista por carrera":

    st.subheader("Lista de aspirantes por carrera")

    carreras = sorted(df_general["Carrera"].dropna().unique())

    carrera_seleccionada = st.selectbox(
        "Selecciona una carrera:",
        carreras
    )

    df_carrera = (
        df_general[df_general["Carrera"] == carrera_seleccionada]
        .copy()
    )

    st.metric(
        "Aspirantes registrados",
        len(df_carrera)
    )

    columnas_preferidas = [
        "Matrícula/ID",
        "APELLIDO PATERNO",
        "APELLIDO MATERNO",
        "NOMBRE (S)",
        "Género",
        "Escuela de Procedencia",
        "Promedio Bachillerato",
        "Promedio_normalizado_100",
        "Estatus_promedio",
        "Observaciones"
    ]

    columnas_disponibles = [
        columna for columna in columnas_preferidas
        if columna in df_carrera.columns
    ]

    st.dataframe(
        df_carrera[columnas_disponibles],
        use_container_width=True,
        hide_index=True
    )

    archivo_excel = convertir_excel_descargable(df_carrera)

    st.download_button(
        label="⬇️ Descargar lista de esta carrera",
        data=archivo_excel,
        file_name=f"aspirantes_{carrera_seleccionada}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# SECCIÓN 4 · CALIDAD DE DATOS
# ============================================================

elif seccion == "Calidad de datos":

    st.subheader("Revisión de calidad de datos")

    resumen_promedios = (
        df_general["Estatus_promedio"]
        .value_counts(dropna=False)
        .reset_index()
    )

    resumen_promedios.columns = [
        "Estatus del promedio",
        "Registros"
    ]

    st.markdown("#### Estado de los promedios")
    st.dataframe(resumen_promedios, use_container_width=True)

    datos_dudosos = df_general[
        df_general["Estatus_promedio"]
        .str.contains("Dato dudoso", na=False)
    ].copy()

    st.markdown("#### Registros con promedio dudoso")

    if datos_dudosos.empty:
        st.success("No se detectaron promedios fuera de rango.")
    else:
        columnas_revision = [
            columna for columna in [
                "Carrera",
                "Matrícula/ID",
                "APELLIDO PATERNO",
                "APELLIDO MATERNO",
                "NOMBRE (S)",
                "Promedio Bachillerato",
                "Promedio_normalizado_100",
                "Estatus_promedio"
            ]
            if columna in datos_dudosos.columns
        ]

        st.dataframe(
            datos_dudosos[columnas_revision],
            use_container_width=True,
            hide_index=True
        )

    columna_id = encontrar_columna(
        df_general,
        ["Matrícula/ID", "Matrícula", "ID"]
    )

    if columna_id is not None:
        duplicados = df_general[
            df_general.duplicated(
                subset=[columna_id],
                keep=False
            )
        ].assign(
    _matricula_orden=lambda x: x[columna_id]
    .fillna("")
    .astype(str)
    .str.strip()
).sort_values("_matricula_orden").drop(columns="_matricula_orden")
        st.markdown("#### Matrículas repetidas")

        if duplicados.empty:
            st.success("No se detectaron matrículas repetidas.")
        else:
            st.warning(
                f"Se detectaron {duplicados[columna_id].nunique()} "
                "matrículas repetidas."
            )
            st.dataframe(
                duplicados,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# SECCIÓN 5 · BASE INTEGRADA
# ============================================================

elif seccion == "Base integrada":

    st.subheader("Base consolidada de aspirantes")

    st.caption(
        "Incluye todas las columnas originales de cada hoja, "
        "más las variables agregadas por el módulo."
    )

    st.dataframe(
        df_general,
        use_container_width=True,
        hide_index=True
    )

    archivo_excel = convertir_excel_descargable(df_general)

    st.download_button(
        label="⬇️ Descargar base integrada",
        data=archivo_excel,
        file_name="base_integrada_aspirantes.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )
