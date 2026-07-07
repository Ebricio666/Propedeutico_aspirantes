import io
import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
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
    "Integración de registros, estadística básica y análisis "
    "de calificaciones de aspirantes."
)


# ============================================================
# FUNCIONES DE APOYO
# ============================================================

def limpiar_texto(valor):
    """Convierte texto a minúsculas, sin acentos y sin espacios repetidos."""
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
    Conserva encabezados originales.
    Si hay encabezados duplicados, agrega un sufijo para evitar errores.
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
    Identifica automáticamente la fila de encabezados.
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
    Si no lo encuentra, utiliza el nombre de la pestaña.
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
    Encuentra una columna aunque cambien acentos, mayúsculas o espacios.
    """
    columnas_limpias = {
        limpiar_texto(columna): columna
        for columna in df.columns
    }

    for posible in posibles_nombres:

        posible_limpio = limpiar_texto(posible)

        if posible_limpio in columnas_limpias:
            return columnas_limpias[posible_limpio]

        for columna_limpia, columna_original in columnas_limpias.items():

            if posible_limpio in columna_limpia:
                return columna_original

    return None


def convertir_promedio(valor):
    """
    Normaliza promedios a una escala de 0 a 100.

    0 a 10       -> multiplica por 10.
    Mayor de 10
    y hasta 100  -> conserva el valor.
    Otro valor   -> dato dudoso.
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


def normalizar_sexo(valor):
    """Homologa los registros de sexo o género."""

    if pd.isna(valor):
        return "Sin especificar"

    texto = limpiar_texto(valor)

    if texto in ["hombre", "masculino", "m", "male"]:
        return "Hombre"

    if texto in ["mujer", "femenino", "f", "female"]:
        return "Mujer"

    return "Sin especificar"


def clasificar_rango_promedio(valor):
    """Clasifica el promedio normalizado en rangos tipo semáforo."""

    if pd.isna(valor):
        return "Sin dato"

    if 60 <= valor < 70:
        return "60-69"

    if 70 <= valor < 80:
        return "70-79"

    if 80 <= valor < 90:
        return "80-89"

    if 90 <= valor <= 100:
        return "90-100"

    return "Fuera de rango"


def procesar_hoja(contenido_archivo, nombre_hoja):
    """
    Lee una hoja del Excel, conserva todas sus columnas
    y agrega variables necesarias para el análisis.
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

    # Elimina únicamente filas completamente vacías.
    df = df.dropna(how="all").copy()

    columna_id = encontrar_columna(
        df,
        ["Matrícula/ID", "Matrícula", "ID"]
    )

    # Conserva solo filas con un identificador.
    if columna_id is not None:
        df = df[df[columna_id].notna()].copy()

    # Variables agregadas sin borrar encabezados originales.
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
    Lee todas las hojas del archivo y genera un único DataFrame.
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
        "Calificaciones por sexo",
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
# PROCESAMIENTO GENERAL DEL ARCHIVO
# ============================================================

contenido_archivo = archivo_subido.getvalue()

with st.spinner("Leyendo e integrando hojas del archivo..."):
    df_general, df_bitacora = procesar_archivo_excel(
        contenido_archivo
    )

if df_general.empty:
    st.error("No se pudieron identificar registros de aspirantes.")
    st.dataframe(df_bitacora, use_container_width=True)
    st.stop()


# ============================================================
# VARIABLES DERIVADAS PARA LOS ANÁLISIS
# ============================================================

columna_sexo = encontrar_columna(
    df_general,
    ["Género", "Genero", "Sexo"]
)

if columna_sexo is not None:
    df_general["Sexo_normalizado"] = df_general[columna_sexo].apply(
        normalizar_sexo
    )
else:
    df_general["Sexo_normalizado"] = "Sin especificar"

df_general["Rango_promedio"] = df_general[
    "Promedio_normalizado_100"
].apply(clasificar_rango_promedio)


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
# SECCIÓN 1 · PANORAMA GENERAL
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
# SECCIÓN 2 · CALIFICACIONES POR SEXO
# ============================================================

elif seccion == "Calificaciones por sexo":

    st.subheader("Distribución de calificaciones por sexo")

    st.caption(
        "Cada barra representa el 100% de aspirantes de ese sexo. "
        "La distribución permite comparar la concentración de promedios "
        "sin que afecte el tamaño distinto de cada grupo."
    )

    orden_rangos = [
        "60-69",
        "70-79",
        "80-89",
        "90-100"
    ]

    orden_sexo = [
        "Mujer",
        "Hombre",
        "Sin especificar"
    ]

    df_calificaciones = df_general[
        df_general["Rango_promedio"].isin(orden_rangos)
    ].copy()

    if df_calificaciones.empty:
        st.warning(
            "No se encontraron promedios válidos entre 60 y 100."
        )
        st.stop()

    tabla_sexo_promedio = (
        df_calificaciones
        .groupby(["Sexo_normalizado", "Rango_promedio"])
        .size()
        .reset_index(name="Aspirantes")
    )

    tabla_sexo_promedio["Rango_promedio"] = pd.Categorical(
        tabla_sexo_promedio["Rango_promedio"],
        categories=orden_rangos,
        ordered=True
    )

    tabla_sexo_promedio["Sexo_normalizado"] = pd.Categorical(
        tabla_sexo_promedio["Sexo_normalizado"],
        categories=orden_sexo,
        ordered=True
    )

    tabla_sexo_promedio = tabla_sexo_promedio.sort_values(
        ["Sexo_normalizado", "Rango_promedio"]
    )

    tabla_sexo_promedio["Porcentaje"] = (
        tabla_sexo_promedio
        .groupby("Sexo_normalizado")["Aspirantes"]
        .transform(lambda x: (x / x.sum()) * 100)
    )

    # Muestra etiquetas solo en segmentos visibles.
    tabla_sexo_promedio["Etiqueta"] = tabla_sexo_promedio[
        "Porcentaje"
    ].apply(
        lambda x: f"{x:.1f}%" if x >= 5 else ""
    )

    fig_barras = px.bar(
        tabla_sexo_promedio,
        x="Porcentaje",
        y="Sexo_normalizado",
        color="Rango_promedio",
        orientation="h",
        barmode="stack",
        text="Etiqueta",
        custom_data=["Aspirantes"],
        category_orders={
            "Sexo_normalizado": orden_sexo,
            "Rango_promedio": orden_rangos
        },
        color_discrete_map={
            "60-69": "#E74C3C",
            "70-79": "#F39C12",
            "80-89": "#F1C40F",
            "90-100": "#27AE60"
        },
        labels={
            "Sexo_normalizado": "Sexo",
            "Porcentaje": "Porcentaje de aspirantes",
            "Rango_promedio": "Rango de promedio"
        }
    )

    fig_barras.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=(
            "<b>Sexo:</b> %{y}<br>"
            "<b>Rango:</b> %{fullData.name}<br>"
            "<b>Aspirantes:</b> %{customdata[0]}<br>"
            "<b>Porcentaje:</b> %{x:.1f}%"
            "<extra></extra>"
        )
    )

    fig_barras.update_layout(
        title="Distribución porcentual de calificaciones por sexo",
        legend_title_text="Rango de promedio",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            title="Porcentaje de aspirantes",
            range=[0, 100],
            ticksuffix="%"
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=[
                "Sin especificar",
                "Hombre",
                "Mujer"
            ]
        ),
        margin=dict(t=100, b=40, l=40, r=40),
        height=420
    )

    st.plotly_chart(
        fig_barras,
        use_container_width=True
    )

    st.markdown("### Cantidad de aspirantes por rango de calificación")

    tabla_conteo = (
        tabla_sexo_promedio
        .pivot(
            index="Sexo_normalizado",
            columns="Rango_promedio",
            values="Aspirantes"
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    for rango in orden_rangos:
        if rango not in tabla_conteo.columns:
            tabla_conteo[rango] = 0

    tabla_conteo["Total"] = tabla_conteo[
        orden_rangos
    ].sum(axis=1)

    tabla_conteo = tabla_conteo[
        tabla_conteo["Total"] > 0
    ].copy()

    tabla_conteo = tabla_conteo[
        ["Sexo_normalizado"] + orden_rangos + ["Total"]
    ]

    tabla_conteo = tabla_conteo.rename(
        columns={
            "Sexo_normalizado": "Sexo",
            "60-69": "🔴 60-69",
            "70-79": "🟠 70-79",
            "80-89": "🟡 80-89",
            "90-100": "🟢 90-100"
        }
    )

    st.dataframe(
        tabla_conteo,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SECCIÓN 3 · LISTA POR CARRERA
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

    df_carrera = df_general[
        df_general["Carrera"] == carrera_seleccionada
    ].copy()

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
        "Sexo",
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
# SECCIÓN 4 · CALIDAD DE DATOS
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

        st.markdown("#### Calificaciones dudosas")

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
# SECCIÓN 5 · BASE INTEGRADA
# ============================================================

elif seccion == "Base integrada":

    st.subheader("Base integrada de aspirantes")

    st.caption(
        "La base incluye todos los registros de las hojas del archivo "
        "en un solo DataFrame. Se preservan los encabezados originales "
        "y se agregan variables para el análisis."
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
# SECCIÓN 6 · BITÁCORA DE CARGA
# ============================================================

elif seccion == "Bitácora de carga":

    st.subheader("Bitácora de lectura del archivo")

    st.dataframe(
        df_bitacora,
        use_container_width=True,
        hide_index=True
    )
