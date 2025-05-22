import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import prince
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

st.title("📊 Dashboard Interactivo: Ventas de Tienda de Conveniencia")

uploaded_file = st.file_uploader("Carga el archivo de datos (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    seccion = st.radio("Selecciona la sección del dashboard:", ["🧭 Análisis General", "📌 Perfil del Cliente y Segmentación"])

    if seccion == "🧭 Análisis General":
        tabs = st.tabs([
            "📁 Exploración General",
            "📈 Evolución de Ventas",
            "🏬 Ventas por Sucursal",
            "💰 Ingreso Bruto",
            "📊 Total, Calificación y Costo"
        ])

        with tabs[0]:
            st.subheader("Vista Previa del Dataset")
            st.dataframe(df.head())
            st.subheader("Resumen Estadístico")
            st.write(df.describe())
            st.subheader("Columnas y Tipos")
            st.write(df.dtypes)
            st.subheader("Valores únicos por columnas seleccionadas")
            columnas_interes = ['Branch', 'Customer type', 'Gender', 'Product line']
            for col in columnas_interes:
                if col in df.columns:
                    st.markdown(f"- **{col}**: {df[col].unique()}")

        with tabs[1]:
            st.subheader("¿Cuál es la evolución de las ventas en el tiempo?")
            ventas_diarias = df.groupby('Date')['Total'].sum().reset_index()
            ventas_diarias['Date_ordinal'] = ventas_diarias['Date'].map(pd.Timestamp.toordinal)
            z = np.polyfit(ventas_diarias['Date_ordinal'], ventas_diarias['Total'], 1)
            p = np.poly1d(z)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ventas_diarias['Date'], y=ventas_diarias['Total'], mode='lines+markers', name='Ventas Diarias'))
            fig.add_trace(go.Scatter(x=ventas_diarias['Date'], y=p(ventas_diarias['Date_ordinal']), mode='lines', name='Tendencia', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.subheader("¿Cuál es la sucursal con más ventas?")
            ventas_sucursal = df.groupby('Branch')['Total'].sum().reset_index()
            fig = px.bar(ventas_sucursal, x='Branch', y='Total', color='Branch', title='Ventas por Sucursal')
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.subheader("¿Hay una diferencia entre el ingreso bruto, las líneas de producto y las sucursales?")
            df_gross = df.groupby(['Branch', 'Product line'])['gross income'].sum().reset_index()
            fig = px.bar(
                df_gross,
                x='Branch',
                y='gross income',
                color='Product line',
                barmode='group',
                text_auto='.2s',
                title='Contribución del Ingreso Bruto por Línea de Producto en cada Sucursal',
                labels={'gross income': 'Ingreso Bruto Total', 'Branch': 'Sucursal'}
            )
            fig.update_layout(
                legend_title_text='Línea de Producto',
                legend=dict(x=1.05, y=1, orientation='v'),
                xaxis_title='Sucursal',
                yaxis_title='Ingreso Bruto Total'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            st.subheader("¿Qué relación existe entre el total, la calificación y el costo unitario?")
            fig = px.scatter_3d(df, x='Unit price', y='Rating', z='Total', color='Total')
            st.plotly_chart(fig, use_container_width=True)

    elif seccion == "📌 Perfil del Cliente y Segmentación":
        tabs = st.tabs([
            "🌟 Sucursal Mejor Evaluada",
            "💳 Métodos de Pago",
            "📦 Línea de Productos",
            "⭐ Calificación del Cliente",
            "👥 Perfil del Cliente",
            "🤖 Clustering y FAMD"
        ])

        with tabs[0]:
            st.subheader("¿Cuál sucursal es la mejor evaluada?")
            avg_rating = df.groupby('Branch')['Rating'].mean().reset_index()
            fig = px.bar(avg_rating, x='Branch', y='Rating', color='Branch', title='Calificación Promedio por Sucursal')
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.subheader("¿Qué método de pago es preferido?")
            fig = px.pie(df, names='Payment', title='Métodos de Pago Preferidos')
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.subheader("¿Qué línea de productos vende más y cuál menos?")
            ventas_producto = df.groupby('Product line')['Total'].sum().reset_index()
            fig = px.bar(ventas_producto, x='Product line', y='Total', color='Total', title='Ventas por Línea de Producto')
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.subheader("¿Qué se relaciona con la calificación de los clientes?")
            # 1. Distribución de calificación por línea de producto
            fig1 = px.box(
            df, x='Product line', y='Rating', color='Product line',
            title='Distribución de Calificación por Línea de Producto'
            )
            st.plotly_chart(fig1, use_container_width=True)

            # 2. Histograma del Rating
            fig2 = px.histogram(
            df, x='Rating', nbins=20, color_discrete_sequence=['indianred'],
            title='Distribución de la Calificación de Clientes'
            )
            st.plotly_chart(fig2, use_container_width=True)

            # 3. Matriz de correlación triangular inferior
            num_vars = ['Unit price', 'Quantity', 'Total', 'Rating', 'Tax 5%', 'gross income']
            corr = df[num_vars].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))

            fig3 = go.Figure(data=go.Heatmap(
            z=corr.mask(mask),
            x=num_vars,
            y=num_vars,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr.round(2).astype(str),
            texttemplate="%{text}"
            ))
            fig3.update_layout(
            title='Matriz de correlación',
            xaxis_title='Variables',
            yaxis_title='Variables'
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 4. Boxplot Rating por tipo de cliente
            fig4 = px.box(
            df, x='Customer type', y='Rating', color='Customer type',
            title='Distribución del Rating por Tipo de Cliente'
            )
            st.plotly_chart(fig4, use_container_width=True)

            # 5. Boxplot Rating por género
            fig5 = px.box(
            df, x='Gender', y='Rating', color='Gender',
            title='Distribución de Calificaciones por Género'
            )
            st.plotly_chart(fig5, use_container_width=True)

            # 6. Boxplot Rating por método de pago y tipo de cliente
            fig6 = px.box(
            df, x='Payment', y='Rating', color='Customer type',
            title='Calificaciones por Método de Pago y Tipo de Cliente'
            )
            st.plotly_chart(fig6, use_container_width=True)

        with tabs[4]:
            st.subheader("¿Qué perfil de cliente se puede apreciar?")
            # 1. Género por tipo de cliente
            fig1 = px.histogram(df, x='Gender', color='Customer type', barmode='group', title='Género por Tipo de Cliente')
            st.plotly_chart(fig1, use_container_width=True)

            # 2. Producto por tipo de cliente
            fig2 = px.histogram(df, x='Product line', color='Customer type', barmode='group', title='Producto por Tipo de Cliente')
            st.plotly_chart(fig2, use_container_width=True)

            # 3. Distribución del gasto total por tipo de cliente (Boxplot)
            fig3 = px.box(df, x='Customer type', y='Total', color='Customer type', title='Distribución del Gasto Total por Tipo de Cliente')
            st.plotly_chart(fig3, use_container_width=True)

            # 4. Densidad del gasto total por tipo de cliente (Violin plot)
            fig4 = px.violin(df, x='Customer type', y='Total', color='Customer type', box=True, points='all',
                     title='Densidad del Gasto Total por Tipo de Cliente')
            fig4.add_shape(type="line", x0=-0.5, x1=2.5, y0=df['Total'].mean(), y1=df['Total'].mean(),
                   line=dict(color='red', dash='dash'), xref='x', yref='y')
            st.plotly_chart(fig4, use_container_width=True)

            # 5. Frecuencia de ventas por líneas de producto y género
            fig5 = px.histogram(df, x='Product line', color='Gender', barmode='group', title='Frecuencia de ventas por Línea de Producto y Género')
            st.plotly_chart(fig5, use_container_width=True)

            # 6. Total por método de pago y tipo de cliente
            fig6 = px.box(df, x='Payment', y='Total', color='Customer type', title='Total de Ventas por Método de Pago y Tipo de Cliente')
            st.plotly_chart(fig6, use_container_width=True)

            # 7. Total por género y tipo de cliente
            fig7 = px.box(df, x='Gender', y='Total', color='Customer type', title='Total de Ventas por Género y Tipo de Cliente')
            st.plotly_chart(fig7, use_container_width=True)

            # 8. Total por género y método de pago con conteo de casos
            fig8 = px.box(df, x='Gender', y='Total', color='Payment', title='Total por Género y Método de Pago')
            # Añadir anotaciones con conteos
            counts = df.groupby(['Gender', 'Payment']).size().reset_index(name='n')
            for _, row in counts.iterrows():
                fig8.add_annotation(
                    x=row['Gender'],
                    y=0,  # parte baja del gráfico
                    text=f"n={row['n']}",
                    showarrow=False,
                    yanchor="bottom",
                    bgcolor="white",
                    font=dict(size=10),
                    xanchor="center"
                )
            st.plotly_chart(fig8, use_container_width=True)

            # 9. Total por sucursal y tipo de cliente
            fig9 = px.box(df, x='Branch', y='Total', color='Customer type', title='Total por Sucursal y Tipo de Cliente')
            st.plotly_chart(fig9, use_container_width=True)

        with tabs[5]:
            st.subheader("Visualización de Aprendizaje No Supervisado (FAMD + Clusters)")

            # Columnas
            cat_cols = ['Gender', 'Customer type', 'Product line', 'Payment', 'Branch']
            num_cols = ['Unit price', 'Total', 'Rating']

            df_famd = df[cat_cols + num_cols].dropna().copy()

            # FAMD
            famd = prince.FAMD(n_components=2, random_state=42)
            famd = famd.fit(df_famd)
            famd_components = famd.transform(df_famd)

            # Limpiar NaNs, asignar nombres correctos
            famd_df = pd.DataFrame(famd_components.values, columns=['F1', 'F2'], index=df_famd.index).dropna()

            # Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            famd_df['cluster'] = kmeans.fit_predict(famd_df)

            # Visualización
            fig = px.scatter(famd_df, x='F1', y='F2', color=famd_df['cluster'].astype(str),
                     title='Clusters en Espacio FAMD (2D)')
            st.plotly_chart(fig, use_container_width=True)
