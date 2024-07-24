# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:24:47 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Función para cargar datos de una hoja específica y convertir 'Fecha' a datetime
@st.cache_data
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    # Verificar si 'Fecha' está en formato texto y convertirlo a datetime
    if pd.api.types.is_string_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', format='%Y-%m-%d')  # Convertir a datetime
    
    return df

def main():
    st.sidebar.title("Ayuda")
    st.sidebar.info("""
        Esta aplicación permite cargar y analizar datos de traspasos de cuentas administradas por las AFORE. 
        Siga estos pasos:

        1. **Cargar archivo xlsx:** Utilice el botón para cargar el archivo Excel.
        2. **Seleccionar Hoja:** Elija la hoja deseada del archivo cargado.
        3. **Filtrar Datos:** Seleccione una descripción del concepto para filtrar los datos.
        4. **Ver Datos y Estadísticas:** Visualice los datos filtrados y sus estadísticas descriptivas.
        5. **Proyección de Datos:** Seleccione un modelo (SARIMA o Suavizado Exponencial) para proyectar datos futuros.
        6. **Ver Proyecciones:** Visualice los datos originales y las proyecciones en gráficos interactivos.
    """)

    st.title('Traspasos: Cuentas Administradas por las AFORE')

    # Ruta del archivo xlsx
    file_path = 'traspasos.xlsx'
    
    # Obtener el nombre de las hojas
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names

    # Selección de la hoja
    selected_sheet = st.selectbox('Selecciona una Hoja', sheet_names)

    # Cargar los datos de la hoja seleccionada
    df = load_data(file_path, selected_sheet)

    # Mostrar una tabla con los datos cargados
    st.subheader('Datos Originales')
    st.write(df)  # Mostrar el DataFrame completo para verificar la columna 'Fecha'

    # Filtrar por Descripción del Concepto
    concepto = st.selectbox('Selecciona una Descripción del Concepto', df['Descripción del Concepto'].unique())

    filtered_df = df[df['Descripción del Concepto'] == concepto]

    # Mostrar los resultados filtrados y estadísticas descriptivas
    st.subheader(f'Datos Filtrados para {concepto}')
    st.write(filtered_df)

    if not filtered_df.empty:
        # Calcular estadísticas descriptivas
        stats = filtered_df['Datos'].describe()

        st.subheader('Estadísticas Descriptivas')
        st.write(stats)

        # Generar gráfico interactivo con Plotly para datos originales
        fig_original = px.line(filtered_df, x='Fecha', y='Datos', title=f'Datos Originales para {concepto}')
        fig_original.update_yaxes(tickprefix='', tickformat='')  # Eliminar el prefijo 'M' en el eje Y
        st.plotly_chart(fig_original)

        # Selección de modelo de predicción
        st.subheader('Proyección de Datos')

        model_option = st.selectbox('Selecciona un modelo de predicción', ['SARIMA', 'Suavizado Exponencial'])

        if model_option == 'SARIMA':
            # Preparar datos para SARIMA
            dates = pd.to_datetime(filtered_df['Fecha'])
            data = filtered_df['Datos'].values

            # Ajustar el modelo SARIMA
            model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit(disp=False)

            # Generar proyección por cada mes durante 3 años
            future_dates = pd.date_range(dates.iloc[-1], periods=36, freq='MS')
            future_preds = model_fit.get_forecast(steps=36).predicted_mean

            # Crear DataFrame para las proyecciones
            future_df = pd.DataFrame({
                'Fecha': future_dates,
                'Datos': future_preds
            })

            st.write('Proyecciones Mensuales Futuras para los próximos 3 años (SARIMA):')
            st.write(future_df)

        elif model_option == 'Suavizado Exponencial':
            # Preparar datos para el suavizado exponencial
            dates = pd.to_datetime(filtered_df['Fecha'])
            data = filtered_df['Datos'].values

            # Ajustar el modelo de suavizado exponencial
            model = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add').fit()

            # Generar proyección por cada mes durante 3 años
            future_dates = pd.date_range(dates.iloc[-1], periods=36, freq='MS')
            future_preds = model.forecast(36)

            # Crear DataFrame para las proyecciones
            future_df = pd.DataFrame({
                'Fecha': future_dates,
                'Datos': future_preds
            })

            st.write('Proyecciones Mensuales Futuras para los próximos 3 años (Suavizado Exponencial):')
            st.write(future_df)

        # Combinar datos originales y proyectados
        combined_df = pd.concat([filtered_df[['Fecha', 'Datos']], future_df])

        # Generar gráfico interactivo con Plotly para datos originales y proyecciones
        fig_combined = go.Figure()

        # Agregar datos originales
        fig_combined.add_trace(go.Scatter(x=filtered_df['Fecha'], y=filtered_df['Datos'], mode='lines', name='Datos Originales'))

        # Agregar datos proyectados
        fig_combined.add_trace(go.Scatter(x=future_df['Fecha'], y=future_df['Datos'], mode='lines', name='Proyección', line=dict(dash='dash', color='red')))

        fig_combined.update_layout(title=f'Datos Originales y Proyección para {concepto}', xaxis_title='Fecha', yaxis_title='Datos')
        fig_combined.update_yaxes(tickprefix='', tickformat='')  # Eliminar el prefijo 'M' en el eje Y

        st.plotly_chart(fig_combined)

    else:
        st.write(f"No hay datos disponibles para la Descripción del Concepto '{concepto}'.")


# Aviso de derechos de autor
#st.sidebar.markdown("""
#    ---
#    © 2024. Todos los derechos reservados.
#    Creado por jahoperi.
#""")

# Pie de página en la barra lateral
st.sidebar.write("© 2024 Todos los derechos reservados")
st.sidebar.write("© 2024 Creado por: Javier Horacio Pérez Ricárdez")
st.sidebar.write("PensionISSSTE: Analista UEAP B")

if __name__ == '__main__':
    main()
