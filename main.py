import io
import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Historial de Aspirantes",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Módulo 1 · Historial de Aspirantes")
st.caption(
    "Integración de registros de aspirantes, estadística básica "
    "y distribución por carrera."
)


# ============================================================
# FUNCIONES DE APOYO
# ============================================================

def limpiar_texto(valor):
    """Convierte texto a minúsculas, sin acentos y espacios repetidos."""
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return re.sub(r"\s+", " ", texto)


def nombres_unicos(encabezados):
    """
    Conserva los encabezados originales.
    Cuando existen columnas repetidas, añade un sufijo para evitar errores.
    """
    usados = {}
    resultado = []

    for posicion, encabezado in enumerate(encabezados, start=1):
        if pd.isna(encabezado) or str(encabezado).strip() == "":
            nombre = f"Columna_sin_nombre_{posicion}"
        else:
            nombre = str(encabezado).strip()

        if nombre in usados:
            usados[nombre] += 1
            nombre = f"{nombre}_{usados[nombre]}"
        else:
            usados[nombre] = 1

        resultado.append(nombre)

    return resultado


def buscar_fila_encabezados(df_crudo):
    """
    Busca la fila donde están los encabezados.
    Funciona aunque cada carrera tenga una fila de encabezados distinta.
    """
    palabras_clave = [
        "matricula/id",
        "matricula",
        "id",
        "apellido paterno",
        "apellido materno",
        "nombre (s)",
        "nombre"
    ]

    limite = min(len(df_crudo), 40)

    for indice in range(limite):
        fila = df_crudo.iloc[indice].tolist()
        valores = [limpiar_texto(valor) for valor in fila]

        coincidencias = sum(
            any(palabra in valor for valor in valores)
            for palabra in palabras_clave
        )

        if coincidencias >= 2:
            return indice

    return None


def obtener_nombre_carrera(nombre_hoja, df_crudo):
    """
    Busca el nombre de carrera dentro de la hoja.
    Si no aparece, utiliza el nombre de la pestaña como respaldo.
    """
    limite = min(len(df_crudo), 15)

    for indice in range(limite):
        fila = df_crudo.iloc[indice].tolist()

        for posicion, valor in enumerate(fila):
            if limpiar_texto(valor) == "carrera":
                if posicion + 1 < len(fila):
                    posible_carrera = fila[posicion + 1]

                    if pd.notna(posible_carrera):
                        return str(posible_carrera).strip()

    return str(nombre_hoja).strip()


def encontrar_columna(df, posibles_nombres):
    """
    Encuentra una columna aunque cambien mayúsculas, acentos o espacios.
    """
    columnas_limpias = {
        limpiar_texto(columna): columna
        for columna in df.columns
    }

    for posible in posibles_nombres:
        posible_limpio = limpiar_texto(posible)

        # Coincidencia exacta
        if posible_limpio in columnas_limpias:
            return columnas_limpias[posible_limpio]

        # Coincidencia parcial
        for columna_limpia, columna_original in columnas_limpias.items():
            if posible_limpio in columna_limpia:
                return columna_original

    return None


def convertir_promedio(valor):
    """
    Convierte promedios a una escala de 0 a 100.

    Regla:
    0 a 10     -> se multiplica por 10.
    10 a 100   -> se conserva.
    Fecha      -> se marca como dudosa.
    Otro valor -> se marca como dudoso.
    """

    if pd.isna(valor) or str(valor).strip() == "":
        return np.nan, "Sin dato"

    if isinstance(valor, (datetime, date, pd.Timestamp)):
        return np.nan, "Dato dudoso: formato fecha"

    texto = str(valor).strip()
    texto = texto.replace("\xa0", " ")
    texto = texto.replace(",", ".")
    texto = texto.lstrip("'").strip()

    try:
        numero = float(texto)
    except (TypeError, ValueError):
        return np.nan, "Dato dudoso: no numérico"

    if 0 <= numero <= 10:
        return round(numero * 10, 2), "Convertido de escala 0-10"

    if 10 < numero <= 100:
        return round(numero, 2), "Válido: escala 0-100"

    return np.nan, "Dato dudoso: fuera de rango"


def procesar_hoja(contenido_archivo, nombre_hoja):
    """
    Lee una hoja, conserva todos sus encabezados y añade Carrera.
    """

    archivo = io.BytesIO(contenido_archivo)

    df_crudo = pd.read_excel(
        archivo,
        sheet_name=nombre_hoja,
        header=None,
        dtype=object
    )

    fila_encabezados = buscar_fila_encabezados(df_crudo)

    if fila_encabezados is None:
        return None, {
            "Hoja": nombre_hoja,
            "Estatus": "No procesada",
            "Detalle": "No se identificó una fila de encabezados."
        }

    carrera = obtener_nombre_carrera(nombre_hoja, df_crudo)

    encabezados = df_crudo.iloc[fila_encabezados].tolist()
    encabezados = nombres_unicos(encabezados)

    df = df_crudo.iloc[fila_encabezados + 1:].copy()
    df.columns = encabezados

    # Elimina solo filas completamente vacías.
    df = df.dropna(how="all").copy()

    columna_id = encontrar_columna(
        df,
        ["Matrícula/ID", "Matrícula", "ID"]
    )

    # Conserva registros con matrícula o ID.
    if columna_id is not None:
        df = df[df[columna_id].notna()].copy()

    # Columnas nuevas para trazabilidad.
    df["Carrera"] = carrera
    df["Hoja_origen"] = nombre_hoja

    columna_promedio = encontrar_columna(
        df,
        [
            "Promedio Bachillerato",
            "Promedio de Bachillerato",
            "Promedio"
        ]
    )

    if columna_promedio is not None:
        df["Promedio_original"] = df[columna_promedio]

        resultado = df[columna_promedio].apply(convertir_promedio)

        df["Promedio_normalizado_100"] = resultado.apply(
            lambda x: x[0]
        )

        df["Estatus_promedio"] = resultado.apply(
            lambda x: x[1]
        )

    else:
        df["Promedio_original"] = np.nan
        df["Promedio_normalizado_100"] = np.nan
        df["Estatus_promedio"] = "No se encontró columna de promedio"

    return df, {
        "Hoja": nombre_hoja,
        "Estatus": "Procesada",
        "Detalle": f"{len(df):,} aspirantes identificados."
    }


@st.cache_data(show_spinner=False)
def procesar_archivo_excel(contenido_archivo):
    """
    Lee todas las hojas y genera un único DataFrame integrado.
    """

    archivo = io.BytesIO(contenido_archivo)
    excel = pd.ExcelFile(archivo)

    bases = []
    bitacora = []

    for hoja in excel.sheet_names:
        df_hoja, resultado = procesar_hoja(
            contenido_archivo,
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
    """Convierte un DataFrame a Excel descargable."""
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
        "Panorama general",
        "Lista por carrera",
        "Calidad de datos",
        "Base integrada",
        "Bitácora de carga"
    ]
)

archivo_subido = st.sidebar.file_uploader(
    "Carga el archivo Excel de aspirantes",
    type=["xlsx", "xls"]
)

if archivo_subido is None:
    st.info("Carga un archivo Excel para iniciar el análisis.")
    st.stop()


# ============================================================
# PROCESAMIENTO
# ============================================================

contenido_archivo = archivo_subido.getvalue()

with st.spinner("Leyendo e integrando las hojas del archivo..."):
    df_general, df_bitacora = procesar_archivo_excel(
        contenido_archivo
    )

if df_general.empty:
    st.error(
        "No se pudieron identificar aspirantes en el archivo cargado."
    )
    st.dataframe(df_bitacora, use_container_width=True)
    st.stop()


# ============================================================
# RESUMEN POR CARRERA
# ============================================================

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

resumen_carrera["Porcentaje"] = (
    resumen_carrera["Aspirantes"]
    / resumen_carrera["Aspirantes"].sum()
    * 100
).round(2)


# ============================================================
# 1. PANORAMA GENERAL
# ============================================================

if seccion == "Panorama general":

    st.subheader("Panorama general de aspirantes")

    total_aspirantes = len(df_general)
    total_carreras = df_general["Carrera"].nunique()
    promedio_general = df_general["Promedio_normalizado_100"].mean()
    mediana_general = df_general["Promedio_normalizado_100"].median()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Aspirantes", f"{total_aspirantes:,}")
    col2.metric("Carreras", total_carreras)
    col3.metric(
        "Promedio general",
        f"{promedio_general:.2f}"
        if pd.notna(promedio_general)
        else "Sin dato"
    )
    col4.metric(
        "Mediana general",
        f"{mediana_general:.2f}"
        if pd.notna(mediana_general)
        else "Sin dato"
    )

    st.markdown("### Participantes por carrera")

    col_tabla, col_grafica = st.columns([1, 1.3])

    with col_tabla:
        st.dataframe(
            resumen_carrera[
                [
                    "Carrera",
                    "Aspirantes",
                    "Porcentaje",
                    "Promedio",
                    "Mediana"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with col_grafica:
        fig_pastel = px.pie(
            resumen_carrera,
            names="Carrera",
            values="Aspirantes",
            hole=0.42
        )

        fig_pastel.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Aspirantes: %{value}<br>"
                "Porcentaje: %{percent}"
                "<extra></extra>"
            )
        )

        fig_pastel.update_layout(
            title="Distribución de aspirantes por carrera",
            legend_title_text="Carrera",
            margin=dict(t=65, b=15, l=15, r=15)
        )

        st.plotly_chart(
            fig_pastel,
            use_container_width=True
        )


# ============================================================
# 2. LISTA POR CARRERA
# ============================================================

elif seccion == "Lista por carrera":

    st.subheader("Lista de aspirantes por carrera")

    carreras = sorted(
        df_general["Carrera"]
        .dropna()
        .astype(str)
        .unique()
    )

    carrera_seleccionada = st.selectbox(
        "Selecciona una carrera:",
        carreras
    )

    df_carrera = (
        df_general[
            df_general["Carrera"] == carrera_seleccionada
        ]
        .copy()
    )

    st.metric("Aspirantes registrados", len(df_carrera))

    columnas_preferidas = [
        "Matrícula/ID",
        "Matrícula",
        "ID",
        "APELLIDO PATERNO",
        "APELLIDO MATERNO",
        "NOMBRE (S)",
        "Nombre",
        "Género",
        "Genero",
        "Escuela de Procedencia",
        "Promedio_original",
        "Promedio_normalizado_100",
        "Estatus_promedio",
        "Observaciones"
    ]

    columnas_disponibles = [
        columna
        for columna in columnas_preferidas
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
# 3. CALIDAD DE DATOS
# ============================================================

elif seccion == "Calidad de datos":

    st.subheader("Revisión de calidad de datos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estado de los promedios")

        resumen_promedios = (
            df_general["Estatus_promedio"]
            .value_counts(dropna=False)
            .reset_index()
        )

        resumen_promedios.columns = [
            "Estatus del promedio",
            "Registros"
        ]

        st.dataframe(
            resumen_promedios,
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("#### Promedios dudosos")

        cantidad_dudosos = df_general[
            df_general["Estatus_promedio"]
            .str.contains("Dato dudoso", na=False)
        ].shape[0]

        st.metric("Registros a revisar", cantidad_dudosos)

    datos_dudosos = df_general[
        df_general["Estatus_promedio"]
        .str.contains("Dato dudoso", na=False)
    ].copy()

    if not datos_dudosos.empty:
        st.markdown("#### Registros con calificación dudosa")

        columnas_revision = [
            columna
            for columna in [
                "Carrera",
                "Matrícula/ID",
                "Matrícula",
                "ID",
                "APELLIDO PATERNO",
                "APELLIDO MATERNO",
                "NOMBRE (S)",
                "Promedio_original",
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

    else:
        st.success("No se detectaron calificaciones dudosas.")

    columna_id = encontrar_columna(
        df_general,
        ["Matrícula/ID", "Matrícula", "ID"]
    )

    if columna_id is not None:
        duplicados = (
            df_general[
                df_general.duplicated(
                    subset=[columna_id],
                    keep=False
                )
            ]
            .assign(
                _orden_id=lambda x: (
                    x[columna_id]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
            )
            .sort_values("_orden_id")
            .drop(columns="_orden_id")
        )

        st.markdown("#### Matrículas o ID repetidos")

        if duplicados.empty:
            st.success("No se detectaron matrículas o ID repetidos.")

        else:
            st.warning(
                f"Se detectaron {duplicados[columna_id].nunique()} "
                "matrículas o ID repetidos."
            )

            st.dataframe(
                duplicados,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 4. BASE INTEGRADA
# ============================================================

elif seccion == "Base integrada":

    st.subheader("Base integrada de aspirantes")

    st.caption(
        "Esta vista une los registros de todas las hojas en un único "
        "DataFrame. Conserva las columnas originales y añade Carrera, "
        "Hoja_origen y variables de normalización del promedio."
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


# ============================================================
# 5. BITÁCORA
# ============================================================

elif seccion == "Bitácora de carga":

    st.subheader("Bitácora de lectura del archivo")

    st.dataframe(
        df_bitacora,
        use_container_width=True,
        hide_index=True
    )
