import io
import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Historial de Aspirantes",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Historial de Aspirantes")
st.caption(
    "Análisis general y análisis por carrera de aspirantes de nuevo ingreso."
)


# ============================================================
# FUNCIONES GENERALES
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


def limpiar_texto_visible(valor):
    """Limpia saltos de línea y espacios conservando mayúsculas."""
    if pd.isna(valor):
        return ""

    texto = str(valor).replace("\n", " ")
    return re.sub(r"\s+", " ", texto).strip()


def nombres_unicos(encabezados):
    """Evita errores cuando existen encabezados repetidos."""
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
                    posible_carrera = fila[posicion + 1]

                    if pd.notna(posible_carrera):
                        return str(posible_carrera).strip()

    return str(nombre_hoja).strip()


def encontrar_columna(df, posibles_nombres):
    """Busca una columna ignorando mayúsculas, acentos y espacios."""

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
    Convierte calificaciones a escala de 0 a 100.

    0 a 10      -> multiplica por 10
    10 a 100    -> conserva
    Otro valor  -> dato dudoso
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
    """Clasifica promedios en rangos tipo semáforo."""

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
# ESTADO DE PROCEDENCIA
# ============================================================

def clasificar_estado_procedencia(valor):
    """
    Clasifica el estado usando el nombre de la institución.
    Cuando no se identifica otro estado, se considera Colima.
    """

    if pd.isna(valor):
        return "Sin dato"

    texto = limpiar_texto(valor)

    if texto in ["", "nan", "none", "escuela de procedencia"]:
        return "Sin dato"

    palabras_jalisco = [
        "jalisco",
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
        "coahuayana",
        "coalcoman",
        "morelia",
        "zamora",
        "lazaro cardenas",
        "uruapan",
        "apatzingan",
        "maravatio"
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

    if "nuevo leon" in texto or "monterrey" in texto:
        return "Nuevo León"

    if "sinaloa" in texto or "culiacan" in texto:
        return "Sinaloa"

    if "durango" in texto:
        return "Durango"

    if "sonora" in texto or "hermosillo" in texto:
        return "Sonora"

    if "baja california" in texto or "tijuana" in texto:
        return "Baja California"

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
# NORMALIZACIÓN DE BACHILLERATOS
# ============================================================

def obtener_numero_institucion(texto, expresiones):
    """Extrae número de plantel cuando existe."""

    for expresion in expresiones:

        coincidencia = re.search(expresion, texto)

        if coincidencia:
            return coincidencia.group(1)

    return None


def normalizar_escuela_procedencia(valor):
    """Agrupa variantes de instituciones similares."""

    if pd.isna(valor):
        return "Sin dato"

    texto_visible = limpiar_texto_visible(valor)
    texto = limpiar_texto(valor)
    texto_compacto = re.sub(r"[^a-z0-9]", "", texto)

    if texto in ["", "nan", "none", "escuela de procedencia"]:
        return "Sin dato"

    if (
        "universidad de colima" in texto
        or "u de c" in texto
        or "udec" in texto
        or "bachillerato udec" in texto
        or re.search(r"\bbachillerato\s*([1-9]|[12][0-9]|30)\b", texto)
    ):
        return "Universidad de Colima (U de C)"

    if (
        "telebachillerato" in texto
        or "tele bachillerato" in texto
        or "telebach" in texto
        or "telebach" in texto_compacto
    ):
        return "Telebachillerato"

    if (
        "colegio de bachilleres" in texto
        or "colegio bachilleres" in texto
        or "colegio de bach" in texto
        or "colegio bach" in texto
        or "cobach" in texto_compacto
        or "coba" in texto_compacto
    ):
        return "Colegio de Bachilleres"

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

    if "cetis" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [r"cetis\s*#?\s*(\d+)"]
        )

        if numero:
            return f"CETis {numero}"

        return "CETis"

    if "cbta" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [r"cbta\s*#?\s*(\d+)"]
        )

        if numero:
            return f"CBTA {numero}"

        return "CBTA"

    if "emsad" in texto_compacto:

        numero = obtener_numero_institucion(
            texto,
            [r"emsad\s*#?\s*(\d+)"]
        )

        if numero:
            return f"EMSAD {numero}"

        return "EMSAD"

    if "isenco" in texto_compacto:
        return "ISENCO"

    if "conalep" in texto_compacto:
        return "CONALEP"

    if "cecyte" in texto_compacto:
        return "CECyTE"

    if "icep" in texto_compacto:
        return "ICEP"

    if (
        "universidad de guadalajara" in texto
        or "udeg" in texto_compacto
        or "prepa regional tuxpan" in texto
    ):
        return "Universidad de Guadalajara (UdeG)"

    if "anahuac" in texto:
        return "Preparatoria Anáhuac"

    if "campoverde" in texto_compacto or "campo verde" in texto:
        return "Colegio Campoverde"

    if "adonai" in texto:
        return "Instituto Adonai"

    if "prepa en linea" in texto:
        return "Prepa en Línea SEP"

    if "acredita" in texto and "bach" in texto:
        return "Acredita-Bach SEP"

    return texto_visible.title()


# ============================================================
# LECTURA DEL EXCEL
# ============================================================

def procesar_hoja(contenido_archivo, nombre_hoja):
    """Lee una hoja y devuelve registros procesados."""

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


# ============================================================
# TABLAS AUXILIARES
# ============================================================

def crear_tabla_calificaciones_por_sexo(df):
    """Prepara barras apiladas de calificaciones por sexo."""

    orden_rangos = ["60-69", "70-79", "80-89", "90-100"]

    df_calificaciones = df[
        df["Rango_promedio"].isin(orden_rangos)
    ].copy()

    if df_calificaciones.empty:
        return pd.DataFrame()

    tabla = (
        df_calificaciones
        .groupby(["Sexo_normalizado", "Rango_promedio"])
        .size()
        .reset_index(name="Aspirantes")
    )

    tabla["Rango_promedio"] = pd.Categorical(
        tabla["Rango_promedio"],
        categories=orden_rangos,
        ordered=True
    )

    tabla["Porcentaje"] = (
        tabla
        .groupby("Sexo_normalizado")["Aspirantes"]
        .transform(lambda x: (x / x.sum()) * 100)
    )

    tabla["Etiqueta"] = tabla["Porcentaje"].apply(
        lambda valor: f"{valor:.1f}%" if valor >= 5 else ""
    )

    return tabla


def crear_distribucion_calificaciones_bachillerato(df, top_n=10):
    """Crea barras apiladas tipo semáforo por bachillerato."""

    orden_rangos = ["60-69", "70-79", "80-89", "90-100"]

    df_valido = df[
        (
            df["Bachillerato_procedencia"] != "Sin dato"
        )
        &
        (
            df["Rango_promedio"].isin(orden_rangos)
        )
    ].copy()

    if df_valido.empty:
        return pd.DataFrame()

    totales = (
        df_valido
        .groupby("Bachillerato_procedencia")
        .size()
        .reset_index(name="Total")
        .sort_values("Total", ascending=False)
    )

    top_bachilleratos = totales.head(top_n).copy()

    escuelas_top = top_bachilleratos[
        "Bachillerato_procedencia"
    ].tolist()

    df_valido = df_valido[
        df_valido["Bachillerato_procedencia"].isin(escuelas_top)
    ].copy()

    tabla = (
        df_valido
        .groupby(
            [
                "Bachillerato_procedencia",
                "Rango_promedio"
            ]
        )
        .size()
        .reset_index(name="Aspirantes")
    )

    tabla = tabla.merge(
        top_bachilleratos,
        on="Bachillerato_procedencia",
        how="left"
    )

    tabla["Porcentaje"] = (
        tabla["Aspirantes"]
        / tabla["Total"]
        * 100
    )

    tabla["Rango_promedio"] = pd.Categorical(
        tabla["Rango_promedio"],
        categories=orden_rangos,
        ordered=True
    )

    tabla["Etiqueta"] = tabla["Porcentaje"].apply(
        lambda valor: f"{valor:.0f}%" if valor >= 8 else ""
    )

    tabla["Escuela_etiqueta"] = tabla.apply(
        lambda fila: (
            f"{fila['Bachillerato_procedencia']} "
            f"(n={int(fila['Total'])})"
        ),
        axis=1
    )

    orden_escuelas = [
        f"{fila['Bachillerato_procedencia']} (n={int(fila['Total'])})"
        for _, fila in top_bachilleratos.iterrows()
    ]

    tabla["Escuela_etiqueta"] = pd.Categorical(
        tabla["Escuela_etiqueta"],
        categories=orden_escuelas[::-1],
        ordered=True
    )

    return tabla


def crear_mapa_colores_carreras(df):
    """Asigna el mismo color a cada carrera en ambos Sunburst."""

    paleta = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Bold
    )

    carreras = sorted(
        df["Carrera"]
        .dropna()
        .astype(str)
        .unique()
    )

    return {
        carrera: paleta[indice % len(paleta)]
        for indice, carrera in enumerate(carreras)
    }


# ============================================================
# GRÁFICAS
# ============================================================

def mostrar_grafica_calificaciones(df):
    """Muestra distribución semáforo de calificaciones por sexo."""

    tabla = crear_tabla_calificaciones_por_sexo(df)

    if tabla.empty:
        st.info("No hay promedios válidos entre 60 y 100.")
        return

    fig = px.bar(
        tabla,
        x="Porcentaje",
        y="Sexo_normalizado",
        color="Rango_promedio",
        orientation="h",
        barmode="stack",
        text="Etiqueta",
        custom_data=["Aspirantes"],
        category_orders={
            "Sexo_normalizado": [
                "Mujer",
                "Hombre",
                "Sin especificar"
            ],
            "Rango_promedio": [
                "60-69",
                "70-79",
                "80-89",
                "90-100"
            ]
        },
        color_discrete_map={
            "60-69": "#E74C3C",
            "70-79": "#F39C12",
            "80-89": "#F1C40F",
            "90-100": "#27AE60"
        }
    )

    fig.update_traces(
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

    fig.update_layout(
        title="Promedio de bachillerato por sexo",
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
        yaxis_title="",
        height=420,
        margin=dict(t=100, b=40, l=30, r=30)
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_grafica_semaforo_bachillerato(df):
    """Muestra distribución semáforo por bachillerato."""

    tabla = crear_distribucion_calificaciones_bachillerato(
        df,
        top_n=10
    )

    if tabla.empty:
        st.info(
            "No hay suficientes promedios válidos para relacionar "
            "con bachilleratos."
        )
        return

    fig = px.bar(
        tabla,
        x="Porcentaje",
        y="Escuela_etiqueta",
        color="Rango_promedio",
        orientation="h",
        barmode="stack",
        text="Etiqueta",
        custom_data=["Aspirantes", "Total"],
        category_orders={
            "Rango_promedio": [
                "60-69",
                "70-79",
                "80-89",
                "90-100"
            ]
        },
        color_discrete_map={
            "60-69": "#E74C3C",
            "70-79": "#F39C12",
            "80-89": "#F1C40F",
            "90-100": "#27AE60"
        }
    )

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=(
            "<b>Bachillerato:</b> %{y}<br>"
            "<b>Rango:</b> %{fullData.name}<br>"
            "<b>Aspirantes:</b> %{customdata[0]} de %{customdata[1]}<br>"
            "<b>Porcentaje:</b> %{x:.1f}%"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title="Distribución de calificaciones por bachillerato",
        legend_title_text="Semáforo",
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
        yaxis_title="",
        height=720,
        margin=dict(t=100, b=40, l=320, r=30)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CONCENTRADO DE ESTADOS
# ============================================================

def mostrar_concentrado_estados(df, max_estados=5):
    """Muestra los principales estados como indicadores compactos."""

    resumen = (
        df[
            df["Estado_procedencia"] != "Sin dato"
        ]
        .groupby("Estado_procedencia")
        .size()
        .reset_index(name="Aspirantes")
        .sort_values("Aspirantes", ascending=False)
    )

    if resumen.empty:
        st.info("No hay información de estado de procedencia.")
        return

    total = resumen["Aspirantes"].sum()

    if len(resumen) > max_estados:

        principales = resumen.head(max_estados - 1).copy()
        otros = resumen.iloc[max_estados - 1:]["Aspirantes"].sum()

        fila_otros = pd.DataFrame({
            "Estado_procedencia": ["Otros estados"],
            "Aspirantes": [otros]
        })

        resumen = pd.concat(
            [principales, fila_otros],
            ignore_index=True
        )

    resumen["Porcentaje"] = (
        resumen["Aspirantes"]
        / total
        * 100
    ).round(1)

    columnas = st.columns(len(resumen))

    for columna, (_, fila) in zip(columnas, resumen.iterrows()):

        columna.metric(
            fila["Estado_procedencia"],
            f"{int(fila['Aspirantes']):,}",
            f"{fila['Porcentaje']:.1f}%",
            delta_color="off"
        )


# ============================================================
# SUNBURST: UDE C -> CARRERAS
# ============================================================

def mostrar_sunburst_udec(df, mapa_colores_carreras):
    """
    Muestra Universidad de Colima -> carreras.

    El porcentaje mostrado en cada carrera es respecto
    al total de aspirantes provenientes de U de C.
    """

    df_udec = df[
        df["Bachillerato_procedencia"]
        == "Universidad de Colima (U de C)"
    ].copy()

    if df_udec.empty:
        st.info("No se encontraron aspirantes provenientes de la U de C.")
        return

    resumen = (
        df_udec
        .groupby("Carrera")
        .size()
        .reset_index(name="Aspirantes")
        .sort_values("Aspirantes", ascending=False)
    )

    total = int(resumen["Aspirantes"].sum())

    labels = ["Universidad de Colima (U de C)"]
    parents = [""]
    values = [total]
    ids = ["root_udec"]
    colores = ["#595959"]
    textos = [f"U de C<br>n={total}"]

    for indice, fila in resumen.reset_index(drop=True).iterrows():

        carrera = fila["Carrera"]
        aspirantes = int(fila["Aspirantes"])
        porcentaje = aspirantes / total * 100

        labels.append(carrera)
        parents.append("root_udec")
        values.append(aspirantes)
        ids.append(f"udec_carrera_{indice}")

        colores.append(
            mapa_colores_carreras.get(carrera, "#9E9E9E")
        )

        textos.append(
            f"{carrera}<br>{porcentaje:.1f}%"
        )

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            text=textos,
            textinfo="text",
            branchvalues="total",
            marker=dict(
                colors=colores,
                line=dict(color="#111217", width=1)
            ),
            insidetextorientation="radial",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Aspirantes: %{value}<br>"
                "Participación: %{percentParent:.1%}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Universidad de Colima → carreras elegidas",
        height=560,
        margin=dict(t=70, b=20, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SUNBURST: OTROS BACHILLERATOS -> CARRERAS
# ============================================================

def mostrar_sunburst_otros_bachilleratos(
    df,
    mapa_colores_carreras,
    top_n=10
):
    """
    Instituciones en escala de grises.
    Carreras con colores fijos.

    El porcentaje de una institución es respecto a todos los
    aspirantes externos. El porcentaje de carrera es respecto
    a cada institución de procedencia.
    """

    df_otros = df[
        (
            df["Bachillerato_procedencia"]
            != "Universidad de Colima (U de C)"
        )
        &
        (
            df["Bachillerato_procedencia"] != "Sin dato"
        )
    ].copy()

    if df_otros.empty:
        st.info("No hay registros de otros bachilleratos.")
        return

    totales = (
        df_otros
        .groupby("Bachillerato_procedencia")
        .size()
        .reset_index(name="Aspirantes")
        .sort_values("Aspirantes", ascending=False)
    )

    escuelas_top = totales.head(top_n)[
        "Bachillerato_procedencia"
    ].tolist()

    df_otros["Escuela_sunburst"] = np.where(
        df_otros["Bachillerato_procedencia"].isin(escuelas_top),
        df_otros["Bachillerato_procedencia"],
        "Otros bachilleratos"
    )

    resumen_escuelas = (
        df_otros
        .groupby("Escuela_sunburst")
        .size()
        .reset_index(name="Aspirantes")
        .sort_values("Aspirantes", ascending=False)
    )

    resumen_carreras = (
        df_otros
        .groupby(["Escuela_sunburst", "Carrera"])
        .size()
        .reset_index(name="Aspirantes")
    )

    total_externos = int(resumen_escuelas["Aspirantes"].sum())

    tonos_gris = [
        "#424242",
        "#505050",
        "#5E5E5E",
        "#6C6C6C",
        "#7A7A7A",
        "#888888",
        "#969696",
        "#A4A4A4",
        "#B2B2B2",
        "#C0C0C0",
        "#6A6A6A"
    ]

    labels = ["Otros bachilleratos"]
    parents = [""]
    values = [total_externos]
    ids = ["root_otros"]
    colores = ["#303030"]
    textos = [f"Otros<br>n={total_externos}"]

    ids_escuelas = {}
    totales_escuelas = {}

    for indice, fila in resumen_escuelas.reset_index(drop=True).iterrows():

        escuela = fila["Escuela_sunburst"]
        aspirantes = int(fila["Aspirantes"])
        porcentaje = aspirantes / total_externos * 100
        id_escuela = f"escuela_{indice}"

        ids_escuelas[escuela] = id_escuela
        totales_escuelas[escuela] = aspirantes

        labels.append(escuela)
        parents.append("root_otros")
        values.append(aspirantes)
        ids.append(id_escuela)

        colores.append(
            tonos_gris[indice % len(tonos_gris)]
        )

        textos.append(
            f"{escuela}<br>{porcentaje:.1f}%"
        )

    for indice, fila in resumen_carreras.reset_index(drop=True).iterrows():

        carrera = fila["Carrera"]
        escuela = fila["Escuela_sunburst"]
        aspirantes = int(fila["Aspirantes"])

        total_escuela = totales_escuelas[escuela]
        porcentaje = aspirantes / total_escuela * 100

        labels.append(carrera)
        parents.append(ids_escuelas[escuela])
        values.append(aspirantes)
        ids.append(f"carrera_{indice}")

        colores.append(
            mapa_colores_carreras.get(carrera, "#9E9E9E")
        )

        textos.append(
            f"{carrera}<br>{porcentaje:.1f}%"
        )

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            text=textos,
            textinfo="text",
            branchvalues="total",
            marker=dict(
                colors=colores,
                line=dict(color="#111217", width=1)
            ),
            insidetextorientation="radial",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Aspirantes: %{value}<br>"
                "Participación: %{percentParent:.1%}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Otros bachilleratos → carreras elegidas",
        height=560,
        margin=dict(t=70, b=20, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CARGA Y PROCESAMIENTO
# ============================================================

archivo_subido = st.sidebar.file_uploader(
    "Carga el archivo Excel de aspirantes",
    type=["xlsx", "xls"]
)

if archivo_subido is None:
    st.info("Carga un archivo Excel para iniciar el análisis.")
    st.stop()

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
# VARIABLES DERIVADAS
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

    df_general["Bachillerato_procedencia"] = df_general[
        columna_escuela
    ].apply(normalizar_escuela_procedencia)

    df_general["Estado_procedencia"] = df_general[
        columna_escuela
    ].apply(clasificar_estado_procedencia)

else:
    df_general["Bachillerato_procedencia"] = "Sin dato"
    df_general["Estado_procedencia"] = "Sin dato"

mapa_colores_carreras = crear_mapa_colores_carreras(df_general)


# ============================================================
# NAVEGACIÓN
# ============================================================

seccion_activa = st.radio(
    "Navegación",
    [
        "📊 Análisis general",
        "🎓 Análisis por carrera"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="navegacion_principal"
)


# ============================================================
# ANÁLISIS GENERAL
# ============================================================

if seccion_activa == "📊 Análisis general":

    st.subheader("Análisis general de aspirantes")

    total = len(df_general)
    mujeres = df_general["Sexo_normalizado"].eq("Mujer").sum()
    hombres = df_general["Sexo_normalizado"].eq("Hombre").sum()
    sin_especificar = df_general["Sexo_normalizado"].eq(
        "Sin especificar"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Aspirantes", f"{total:,}")
    col2.metric("Mujeres", f"{mujeres:,}")
    col3.metric("Hombres", f"{hombres:,}")
    col4.metric("Sin especificar", f"{sin_especificar:,}")

    st.markdown("### Distribución de calificaciones por sexo")
    mostrar_grafica_calificaciones(df_general)

    st.markdown("### Estado de procedencia")
    mostrar_concentrado_estados(df_general)

    st.markdown(
        "## Distribución de calificaciones por bachillerato"
    )
    mostrar_grafica_semaforo_bachillerato(df_general)

    st.markdown("## Origen académico y carrera elegida")

    col_udec, col_otros = st.columns(2)

    with col_udec:
        mostrar_sunburst_udec(
            df_general,
            mapa_colores_carreras
        )

    with col_otros:
        mostrar_sunburst_otros_bachilleratos(
            df_general,
            mapa_colores_carreras,
            top_n=10
        )


# ============================================================
# ANÁLISIS POR CARRERA
# ============================================================

elif seccion_activa == "🎓 Análisis por carrera":

    st.subheader("Análisis por carrera")

    carreras = sorted(
        df_general["Carrera"]
        .dropna()
        .astype(str)
        .unique()
    )

    carrera_seleccionada = st.selectbox(
        "Selecciona una carrera",
        options=carreras,
        key="selector_carrera"
    )

    df_carrera = df_general[
        df_general["Carrera"] == carrera_seleccionada
    ].copy()

    st.markdown(f"## {carrera_seleccionada}")

    total_carrera = len(df_carrera)
    mujeres_carrera = df_carrera["Sexo_normalizado"].eq(
        "Mujer"
    ).sum()
    hombres_carrera = df_carrera["Sexo_normalizado"].eq(
        "Hombre"
    ).sum()
    sin_especificar_carrera = df_carrera["Sexo_normalizado"].eq(
        "Sin especificar"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Aspirantes", f"{total_carrera:,}")
    col2.metric("Mujeres", f"{mujeres_carrera:,}")
    col3.metric("Hombres", f"{hombres_carrera:,}")
    col4.metric(
        "Sin especificar",
        f"{sin_especificar_carrera:,}"
    )

    st.markdown("### Distribución de calificaciones por sexo")
    mostrar_grafica_calificaciones(df_carrera)

    st.markdown("### Estado de procedencia")
    mostrar_concentrado_estados(df_carrera)

    st.markdown(
        "## Distribución de calificaciones por bachillerato"
    )
    mostrar_grafica_semaforo_bachillerato(df_carrera)
