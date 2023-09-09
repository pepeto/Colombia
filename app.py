import pandas as pd
import streamlit as st
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import KDTree

st.title("Aplicación para Análisis Geoespacial")

# Carga de datos
ruta_geojson = st.file_uploader("Por favor suba el archivo GeoJSON", type=["geojson"])
ruta_parquet = st.file_uploader("Por favor suba el archivo Parquet", type=["parquet"])

if ruta_geojson and ruta_parquet:
    geo_df = gpd.read_file(ruta_geojson)
    geometrias_nulas_o_corruptas = geo_df[geo_df['geometry'].isnull()]
    st.write(geometrias_nulas_o_corruptas.head())

    geo_df = geo_df[geo_df['geometry'].notnull()]
    geo_df['bounds'] = geo_df['geometry'].apply(lambda x: x.bounds)
    coordenadas_centrales = geo_df['bounds'].apply(lambda x: ((x[0]+x[2])/2, (x[1]+x[3])/2)).tolist()

    # Creación de KDTree
    kdtree = KDTree(coordenadas_centrales)

    # Caché para resultados
    cache_resultados = {}

    def encontrar_municipio_optimizado(latitud, longitud):
        if pd.isna(latitud) or pd.isna(longitud):
            return "A - Falta Coordenada"
        clave_cache = (latitud, longitud)
        if clave_cache in cache_resultados:
            return cache_resultados[clave_cache]
        punto = Point(longitud, latitud)
        indices_cercanos = kdtree.query([[longitud, latitud]], k=5, return_distance=False)[0]
        for indice in indices_cercanos:
            if geo_df.iloc[indice]['geometry'].contains(punto):
                cache_resultados[clave_cache] = geo_df.iloc[indice]['MPIO_CNMBR']
                return geo_df.iloc[indice]['MPIO_CNMBR']
        for _, municipio in geo_df.iterrows():
            if municipio['geometry'].contains(punto):
                cache_resultados[clave_cache] = municipio['MPIO_CNMBR']
                return municipio['MPIO_CNMBR']
        cache_resultados[clave_cache] = "A - Ningún municipio conocido."
        return "A - Ningún municipio conocido."

    # Leer datos parquet
    df_moova = pd.read_parquet(ruta_parquet)
    df_moova['latitud'] = df_moova['to_geo'].str.extract(r'POINT\(([-\d.]+) ([-\d.]+)\)', expand=True)[1].astype(float)
    df_moova['longitud'] = df_moova['to_geo'].str.extract(r'POINT\(([-\d.]+) ([-\d.]+)\)', expand=True)[0].astype(float)
    df_moova['nombre_municipio'] = df_moova.apply(lambda row: encontrar_municipio_optimizado(row['latitud'], row['longitud']), axis=1)
    
    # Convertimos la columna de geometría a strings para evitar errores de serialización
    geo_df['geometry'] = geo_df['geometry'].apply(lambda x: str(x))

    # Guardar archivo parquet
    salida_parquet = 'salida_temp.parquet'
    df_moova.to_parquet(salida_parquet, index=False)

    st.write("Errores: ", df_moova['nombre_municipio'].str.startswith('A - ').sum())

    # Botón para descargar el archivo de salida
    if st.button('Descargar Archivo de Salida'):
        with open(salida_parquet, 'rb') as f:
            bytes_data = f.read()
            st.download_button(
                label="Descargar Archivo Parquet",
                data=bytes_data,
                file_name='salida.parquet',
                mime='application/octet-stream',
            )