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
    "Integración de registros, análisis de calificaciones y perfil "
    "de procedencia académica de aspirantes."
)


# ============================================================
# FUNCIONES GENERALES
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


def limpiar_texto_visible(valor):
    """Limpia espacios y saltos de línea conservando el texto visible."""
    if pd.isna(valor):
        return ""

    texto = str(valor).replace("\n", " ")
    return re.sub(r"\s+", " ", texto).strip()


def titulo_estandar(valor):
    """Da formato más legible a nombres de instituciones."""
    texto = limpiar_texto_visible(valor)

    if not texto:
        return "Sin dato"

    texto = texto.lower()

    palabras_mayusculas = {
        "cbtis": "CBTis",
        "cetis": "CETis",
        "cbta": "CBTA",
        "emsad": "EMSAD",
        "isenco": "ISENCO",
        "conalep": "CONALEP",
        "icep": "ICEP",
        "cecyte": "CECyTE",
        "udeg": "UdeG",
        "udec": "UdeC",
        "sep": "SEP",
        "cobach": "COBACH",
        "coba": "COBA"
    }

    partes = []

    for palabra in texto.split():
        palabra_limpia = re.sub(r"[^a-z0-9áéíóúñ]", "", palabra)

        if palabra_limpia in palabras_mayusculas:
            partes.append(palabras_mayusculas[palabra_limpia])
        else:
            partes.append(palabra.capitalize())

    return " ".join(partes)


def nombres_unicos(encabezados):
    """Evita errores cuando hay encabezados repetidos."""
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
    """Busca automáticamente la fila donde están los encabezados."""

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

        valores = [
            limpiar_texto(valor)
            for valor in df_crudo.iloc[indice].tolist()
        ]

        coincidencias = sum(
            any(palabra in valor for valor in valores)
            for palabra in palabras_clave
        )

        if coincidencias >= 2:
            return indice

    return None


def obtener_nombre_carrera(nombre_hoja, df_crudo):
    """Busca el nombre de carrera dentro de la hoja."""

    limite = min(len(df_crudo), 15)

    for indice in range(limite):

        fila = df_crudo.iloc[indice].tolist()

        for posicion, valor in enumerate(fila):

            if limpiar_texto(valor) == "carrera":

                if posicion + 1 < len(fila):
                    carrera = fila[posicion + 1]

                    if pd.notna(carrera):
                        return str(carrera).strip()

    return str(nombre_hoja).strip()


def encontrar_columna(df, posibles_nombres):
    """Encuentra columnas aunque cambien acentos, mayúsculas o espacios."""

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


# ============================================================
# CALIFICACIONES
# ============================================================

def convertir_promedio(valor):
    """
    Normaliza calificaciones a escala de 0 a 100.

    0 a 10       → multiplica por 10.
    10 a 100     → conserva.
    Otro valor   → dato dudoso.
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


def clasificar_rango_promedio(valor):
    """Clasifica promedios en rangos semáforo."""

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


# ============================================================
# SEXO
# ============================================================

def normalizar_sexo(valor):
    """Homologa registros de sexo o género."""

    if pd.isna(valor):
        return "Sin especificar"

    texto = limpiar_texto(valor)

    if texto in ["hombre", "masculino", "m", "male"]:
        return "Hombre"

    if texto in ["mujer", "femenino", "f", "female"]:
        return "Mujer"

    return "Sin especificar"


# ============================================================
# PROCEDENCIA ESTATAL
# ============================================================

def clasificar_estado_procedencia(valor):
    """
    Clasifica el estado según el nombre de la escuela.

    Los registros sin coincidencia se consideran Colima, según el
    criterio definido para este proyecto.
    """

    if pd.isna(valor):
        return "Sin dato"

    texto = limpiar_texto(valor)

    if texto in ["", "nan", "none", "escuela de procedencia"]:
        return "Sin dato"

    palabras_jalisco = [
        "jalisco",
        " jal.",
        "tuxpan",
        "cihuatlan",
        "autlan",
        "guadalajara",
        "zapopan",
        "tonala",
        "sayula",
        "zapotiltic",
        "zapotlan",
        "ciudad guzman",
        "tequila",
        "casimiro castillo",
        "el grullo",
        "union de tula",
        "tamazula",
        "teocuitatlan",
        "universidad de guadalajara",
        "udeg"
    ]

    if any(palabra in texto for palabra in palabras_jalisco):
        return "Jalisco"

    palabras_michoacan = [
        "michoacan",
        " mich.",
        "coahuayana",
        "coalcoman",
        "morelia",
        "zamora",
        "lazaro cardenas",
        "uruapan",
        "apatzingan",
        "maravatio",
        "colegio de bachilleres del edo de mich"
    ]

    if any(palabra in texto for palabra in palabras_michoacan):
        return "Michoacán"

    palabras_nayarit = [
        "nayarit",
        "tepic",
        "bahia de banderas",
        "santiago ixcuintla",
        "compostela"
    ]

    if any(palabra in texto for palabra in palabras_nayarit):
        return "Nayarit"

    palabras_guanajuato = [
        "guanajuato",
        "leon",
        "irapuato",
        "celaya",
        "salamanca"
    ]

    if any(palabra in texto for palabra in palabras_guanajuato):
        return "Guanajuato"

    palabras_nuevo_leon = [
        "nuevo leon",
        "monterrey",
        "san nicolas",
        "guadalupe n.l."
    ]

    if any(palabra in texto for palabra in palabras_nuevo_leon):
        return "Nuevo León"

    palabras_sinaloa = [
        "sinaloa",
        "culiacan",
        "mazatlan",
        "los mochis"
    ]

    if any(palabra in texto for palabra in palabras_sinaloa):
        return "Sinaloa"

    if "durango" in texto:
        return "Durango"

    if "sonora" in texto or "hermosillo" in texto:
        return "Sonora"

    if "baja california" in texto or "tijuana" in texto:
        return "Baja California"

    if "quintana roo" in texto:
        return "Quintana Roo"

    if "veracruz" in texto:
        return "Veracruz"

    if "ciudad de mexico" in texto or "cdmx" in texto:
        return "Ciudad de México"

    if any(
        palabra in texto
        for palabra in ["canada", "canadá", "usa", "united states"]
    ):
        return "Internacional"

    return "Colima"


# ============================================================
# NORMALIZACIÓN DE ESCUELAS
# ============================================================

def obtener_numero_institucion(texto, expresiones):
    """Busca el número de plantel con expresiones regulares."""

    for expresion in expresiones:
        coincidencia = re.search(expresion, texto)

        if coincidencia:
            return coincidencia.group(1)

    return None


def obtener_localidad_emsad(texto):
    """Identifica localidad para estandarizar EMSAD."""

    if "minatitlan" in texto:
        return "Minatitlán"

    if "lo de villa" in texto:
        return "Lo de Villa"

    if "zacualpan" in texto:
        return "Zacualpan"

    if "ixtlahuacan" in texto:
        return "Ixtlahuacán"

    if "coquimatlan" in texto:
        return "Coquimatlán"

    if "comala" in texto:
        return "Comala"

    return None


def normalizar_escuela_procedencia(valor):
    """
    Agrupa variantes de una misma escuela.

    Ejemplos:
    CBTis #19 / CBTIS 19 / CBTIs 19 → CBTis 19
    EMSAD / EMSAD #1 / EMSAD Minatitlán → EMSAD 1 · Minatitlán
    Telebachillerato / Telebach / Tele Bach → Telebachillerato
    Colegio de Bachilleres / Colegio de Bach / COBACH → Colegio de Bachilleres
    Bachilleratos UdeC → Universidad de Colima (U de C)
    """

    if pd.isna(valor):
        return "Sin dato"

    escuela_visible = limpiar_texto_visible(valor)
    texto = limpiar_texto(valor)

    if texto in ["", "nan", "none", "escuela de procedencia"]:
        return "Sin dato"

    texto_compacto = re.sub(r"[^a-z0-9]", "", texto)

    # --------------------------------------------------------
    # UNIVERSIDAD DE COLIMA
    # --------------------------------------------------------
    es_udec = (
        "universidad de colima" in texto
        or "u de c" in texto
        or "udec" in texto
        or (
            re.search(r"\bbach\.?\s*\d+", texto) is not None
            and ("colima" in texto or "col." in texto)
        )
    )

    if es_udec:
        return "Universidad de Colima (U de C)"

    # --------------------------------------------------------
    # TELEBACHILLERATO
    # --------------------------------------------------------
    es_telebach = (
        "telebachillerato" in texto
        or "tele bachillerato" in texto
        or "telebach" in texto_compacto
        or "telebach" in texto
    )

    if es_telebach:
        return "Telebachillerato"

    # --------------------------------------------------------
    # COLEGIO DE BACHILLERES
    # --------------------------------------------------------
    es_colegio_bach = (
        "colegio de bachilleres" in texto
        or "colegio bachilleres" in texto
        or "colegio de bach" in texto
        or "colegio bach" in texto
        or "cobach" in texto_compacto
        or "coba" in texto_compacto
    )

    if es_colegio_bach:
        return "Colegio de Bachilleres"

    # --------------------------------------------------------
    # CBTis
    # --------------------------------------------------------
    if "cbtis" in texto_compacto or "cbti" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [
                r"cbtis\s*#?\s*(\d+)",
                r"cbti[s]?\s*#?\s*(\d+)"
            ]
        )

        if numero:
            return f"CBTis {numero}"

        return "CBTis"

    # --------------------------------------------------------
    # CETis
    # --------------------------------------------------------
    if "cetis" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [
                r"cetis\s*#?\s*(\d+)"
            ]
        )

        if numero:
            return f"CETis {numero}"

        return "CETis"

    # --------------------------------------------------------
    # CBTA
    # --------------------------------------------------------
    if "cbta" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [
                r"cbta\s*#?\s*(\d+)"
            ]
        )

        if numero:
            return f"CBTA {numero}"

        return "CBTA"

    # --------------------------------------------------------
    # EMSAD
    # --------------------------------------------------------
    if "emsad" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [
                r"emsad\s*#?\s*(\d+)"
            ]
        )

        localidad = obtener_localidad_emsad(texto)

        if numero and localidad:
            return f"EMSAD {numero} · {localidad}"

        if numero:
            return f"EMSAD {numero}"

        if localidad == "Minatitlán":
            return "EMSAD 1 · Minatitlán"

        if localidad:
            return f"EMSAD · {localidad}"

        return "EMSAD"

    # --------------------------------------------------------
    # ISENCO
    # --------------------------------------------------------
    if "isenco" in texto_compacto:
        return "ISENCO"

    # --------------------------------------------------------
    # CONALEP
    # --------------------------------------------------------
    if "conalep" in texto_compacto:
        return "CONALEP"

    # --------------------------------------------------------
    # CECyTE
    # --------------------------------------------------------
    if "cecyte" in texto_compacto:
        return "CECyTE"

    # --------------------------------------------------------
    # ICEP
    # --------------------------------------------------------
    if "icep" in texto_compacto:
        return "ICEP"

    # --------------------------------------------------------
    # UNIVERSIDAD DE GUADALAJARA
    # --------------------------------------------------------
    if (
        "universidad de guadalajara" in texto
        or "udeg" in texto_compacto
        or "prepa regional tuxpan" in texto
    ):
        return "Universidad de Guadalajara (UdeG)"

    # --------------------------------------------------------
    # PREPARATORIA ANÁHUAC
    # --------------------------------------------------------
    if "anahuac" in texto:
        return "Preparatoria Anáhuac"

    # --------------------------------------------------------
    # COLEGIO CAMPOVERDE
    # --------------------------------------------------------
    if "campoverde" in texto_compacto or "campo verde" in texto:
        return "Colegio Campoverde"

    # --------------------------------------------------------
    # INSTITUTO ADONAI
    # --------------------------------------------------------
    if "adonai" in texto:
        return "Instituto Adonai"

    # --------------------------------------------------------
    # ACREDITA-BACH
    # --------------------------------------------------------
    if "acredita" in texto and "bach" in texto:
        return "Acredita-Bach SEP"

    # --------------------------------------------------------
    # PREPA EN LÍNEA SEP
    # --------------------------------------------------------
    if "prepa en linea" in texto:
        return "Prepa en Línea SEP"

    # --------------------------------------------------------
    # RESTO DE ESCUELAS
    # --------------------------------------------------------
    texto_base = re.sub(r"[^a-z0-9áéíóúñ ]", " ", texto)
    texto_base = re.sub(r"\s+", " ", texto_base).strip()

    return titulo_estandar(texto_base)


def preparar_top_n_con_otros(
    df_resumen,
    columna_categoria,
    columna_valor,
    top_n=10
):
    """
    Conserva las categorías más frecuentes y agrupa el resto como Otros.
    """

    if df_resumen.empty:
        return pd.DataFrame(
            columns=[columna_categoria, columna_valor, "Porcentaje"]
        )

    df_ordenado = df_resumen.sort_values(
        columna_valor,
        ascending=False
    ).copy()

    top = df_ordenado.head(top_n).copy()
    resto = df_ordenado.iloc[top_n:].copy()

    if not resto.empty:
        fila_otros = pd.DataFrame({
            columna_categoria: ["Otros"],
            columna_valor: [resto[columna_valor].sum()]
        })

        top = pd.concat(
            [top, fila_otros],
            ignore_index=True
        )

    top["Porcentaje"] = (
        top[columna_valor]
        / top[columna_valor].sum()
        * 100
    ).round(2)

    return top


# ============================================================
# PROCESAMIENTO DEL ARCHIVO
# ============================================================

def procesar_hoja(contenido_archivo, nombre_hoja):
    """Lee una hoja de Excel y devuelve sus registros procesados."""

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

    encabezados = nombres_unicos(
        df_crudo.iloc[fila_encabezados].tolist()
    )

    df = df_crudo.iloc[fila_encabezados + 1:].copy()
    df.columns = encabezados
    df = df.dropna(how="all").copy()

    columna_id = encontrar_columna(
        df,
        ["Matrícula/ID", "Matrícula", "ID"]
    )

    if columna_id is not None:
        df = df[df[columna_id].notna()].copy()

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
    """Integra todas las hojas del archivo en un solo DataFrame."""

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
    """Convierte un DataFrame en archivo Excel descargable."""

    salida = io.BytesIO()

    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Aspirantes_integrados"
        )

    return salida.getvalue()


# ============================================================
# MENÚ
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
# CARGA Y VARIABLES DERIVADAS
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

columna_escuela = encontrar_columna(
    df_general,
    [
        "Escuela de Procedencia",
        "Escuela Procedencia",
        "Procedencia",
        "Escuela"
    ]
)

if columna_escuela is not None:

    df_general["Escuela_procedencia_original"] = df_general[
        columna_escuela
    ].apply(limpiar_texto_visible)

    df_general["Estado_procedencia"] = df_general[
        columna_escuela
    ].apply(clasificar_estado_procedencia)

    df_general["Escuela_procedencia_categoria"] = df_general[
        columna_escuela
    ].apply(normalizar_escuela_procedencia)

else:
    df_general["Escuela_procedencia_original"] = "Sin dato"
    df_general["Estado_procedencia"] = "Sin dato"
    df_general["Escuela_procedencia_categoria"] = "Sin dato"


# ============================================================
# TABLAS RESUMEN
# ============================================================

resumen_carrera = (
    df_general
    .groupby("Carrera", dropna=False)
    .agg(
        Aspirantes=("Carrera", "size"),
        Promedio=("Promedio_normalizado_100", "mean"),
        Mediana=("Promedio_normalizado_100", "median")
    )
    .reset_index()
    .sort_values("Aspirantes", ascending=False)
)

resumen_carrera["Promedio"] = resumen_carrera["Promedio"].round(2)
resumen_carrera["Mediana"] = resumen_carrera["Mediana"].round(2)

resumen_carrera["Porcentaje"] = (
    resumen_carrera["Aspirantes"]
    / resumen_carrera["Aspirantes"].sum()
    * 100
).round(2)

resumen_estado = (
    df_general[
        df_general["Estado_procedencia"] != "Sin dato"
    ]
    .groupby("Estado_procedencia", dropna=False)
    .size()
    .reset_index(name="Aspirantes")
    .sort_values("Aspirantes", ascending=False)
)

if not resumen_estado.empty:
    resumen_estado["Porcentaje"] = (
        resumen_estado["Aspirantes"]
        / resumen_estado["Aspirantes"].sum()
        * 100
    ).round(2)

resumen_escuela = (
    df_general[
        df_general["Escuela_procedencia_categoria"] != "Sin dato"
    ]
    .groupby("Escuela_procedencia_categoria", dropna=False)
    .size()
    .reset_index(name="Aspirantes")
    .sort_values("Aspirantes", ascending=False)
)

if not resumen_escuela.empty:
    resumen_escuela["Porcentaje"] = (
        resumen_escuela["Aspirantes"]
        / resumen_escuela["Aspirantes"].sum()
        * 100
    ).round(2)

resumen_escuela_visual = preparar_top_n_con_otros(
    resumen_escuela,
    columna_categoria="Escuela_procedencia_categoria",
    columna_valor="Aspirantes",
    top_n=10
)


# ============================================================
# PANORAMA GENERAL
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

    st.markdown("### Perfil general de ingreso")

    col_carrera, col_estado = st.columns(2)

    with col_carrera:

        fig_carrera = px.pie(
            resumen_carrera,
            names="Carrera",
            values="Aspirantes",
            hole=0.45
        )

        fig_carrera.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Aspirantes: %{value}<br>"
                "Porcentaje: %{percent}"
                "<extra></extra>"
            )
        )

        fig_carrera.update_layout(
            title="Aspirantes por carrera",
            legend_title_text="Carrera",
            height=440
        )

        st.plotly_chart(
            fig_carrera,
            use_container_width=True
        )

    with col_estado:

        fig_estado = px.pie(
            resumen_estado,
            names="Estado_procedencia",
            values="Aspirantes",
            hole=0.45
        )

        fig_estado.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Aspirantes: %{value}<br>"
                "Porcentaje: %{percent}"
                "<extra></extra>"
            )
        )

        fig_estado.update_layout(
            title="Procedencia por estado",
            legend_title_text="Estado",
            height=440
        )

        st.plotly_chart(
            fig_estado,
            use_container_width=True
        )

    st.markdown("### Escuela de procedencia")

    if resumen_escuela_visual.empty:
        st.info("No se encontraron escuelas de procedencia registradas.")

    else:
        fig_escuela = px.pie(
            resumen_escuela_visual,
            names="Escuela_procedencia_categoria",
            values="Aspirantes",
            hole=0.45,
            custom_data=["Porcentaje"]
        )

        fig_escuela.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Aspirantes: %{value}<br>"
                "Porcentaje: %{customdata[0]:.1f}%"
                "<extra></extra>"
            )
        )

        fig_escuela.update_layout(
            title=(
                "Top 10 escuelas de procedencia "
                "y agrupación de las restantes"
            ),
            legend_title_text="Escuela",
            height=580
        )

        st.plotly_chart(
            fig_escuela,
            use_container_width=True
        )

    with st.expander("Ver tabla completa de escuelas agrupadas"):

        tabla_escuela = resumen_escuela.rename(
            columns={
                "Escuela_procedencia_categoria": "Escuela"
            }
        )

        st.dataframe(
            tabla_escuela[
                ["Escuela", "Aspirantes", "Porcentaje"]
            ],
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Ver variantes originales agrupadas"):

        variantes = (
            df_general[
                [
                    "Escuela_procedencia_original",
                    "Escuela_procedencia_categoria"
                ]
            ]
            .value_counts()
            .reset_index(name="Aspirantes")
            .sort_values(
                ["Escuela_procedencia_categoria", "Aspirantes"],
                ascending=[True, False]
            )
        )

        variantes = variantes.rename(
            columns={
                "Escuela_procedencia_original": "Registro original",
                "Escuela_procedencia_categoria": "Categoría agrupada"
            }
        )

        st.dataframe(
            variantes,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CALIFICACIONES POR SEXO
# ============================================================

elif seccion == "Calificaciones por sexo":

    st.subheader("Distribución de calificaciones por sexo")

    orden_rangos = ["60-69", "70-79", "80-89", "90-100"]
    orden_sexo = ["Mujer", "Hombre", "Sin especificar"]

    df_calificaciones = df_general[
        df_general["Rango_promedio"].isin(orden_rangos)
    ].copy()

    if df_calificaciones.empty:
        st.warning("No se encontraron promedios válidos entre 60 y 100.")
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

    tabla_sexo_promedio["Porcentaje"] = (
        tabla_sexo_promedio
        .groupby("Sexo_normalizado")["Aspirantes"]
        .transform(lambda x: x / x.sum() * 100)
    )

    tabla_sexo_promedio["Etiqueta"] = tabla_sexo_promedio[
        "Porcentaje"
    ].apply(
        lambda valor: f"{valor:.1f}%" if valor >= 5 else ""
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
        }
    )

    fig_barras.update_traces(
        textposition="inside",
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
        xaxis=dict(
            title="Porcentaje de aspirantes",
            range=[0, 100],
            ticksuffix="%"
        ),
        yaxis_title="",
        height=420
    )

    st.plotly_chart(
        fig_barras,
        use_container_width=True
    )

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
# LISTA POR CARRERA
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
        "Género",
        "Genero",
        "Sexo",
        "Escuela de Procedencia",
        "Escuela_procedencia_categoria",
        "Estado_procedencia",
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
# CALIDAD DE DATOS
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

    st.dataframe(
        resumen_promedios,
        use_container_width=True,
        hide_index=True
    )

    datos_dudosos = df_general[
        df_general["Estatus_promedio"]
        .str.contains("Dato dudoso", na=False)
    ].copy()

    if not datos_dudosos.empty:
        st.markdown("#### Registros con calificación dudosa")
        st.dataframe(
            datos_dudosos,
            use_container_width=True,
            hide_index=True
        )

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
                _orden=lambda x: (
                    x[columna_id]
                    .fillna("")
                    .astype(str)
                )
            )
            .sort_values("_orden")
            .drop(columns="_orden")
        )

        if not duplicados.empty:
            st.markdown("#### Matrículas o ID repetidos")
            st.dataframe(
                duplicados,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# BASE INTEGRADA
# ============================================================

elif seccion == "Base integrada":

    st.subheader("Base integrada de aspirantes")

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
# BITÁCORA
# ============================================================

elif seccion == "Bitácora de carga":

    st.subheader("Bitácora de lectura del archivo")

    st.dataframe(
        df_bitacora,
        use_container_width=True,
        hide_index=True
    )
