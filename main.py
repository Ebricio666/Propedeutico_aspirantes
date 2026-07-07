def crear_resumen_bachillerato(df):
    """Crea Top 10 de bachilleratos y agrupa el resto como Otros."""

    resumen = (
        df[df["Bachillerato_procedencia"] != "Sin dato"]
        .groupby("Bachillerato_procedencia")
        .size()
        .reset_index(name="Aspirantes")
        .sort_values("Aspirantes", ascending=False)
    )

    if resumen.empty:
        return pd.DataFrame()

    top_10 = resumen.head(10).copy()
    restantes = resumen.iloc[10:].copy()

    if not restantes.empty:
        fila_otros = pd.DataFrame({
            "Bachillerato_procedencia": ["Otros"],
            "Aspirantes": [restantes["Aspirantes"].sum()]
        })

        top_10 = pd.concat(
            [top_10, fila_otros],
            ignore_index=True
        )

    top_10["Porcentaje"] = (
        top_10["Aspirantes"]
        / top_10["Aspirantes"].sum()
        * 100
    ).round(2)

    top_10["Etiqueta_bachillerato"] = top_10.apply(
        lambda fila: (
            f"{fila['Bachillerato_procedencia']} · "
            f"n={int(fila['Aspirantes'])}"
        ),
        axis=1
    )

    return top_10
