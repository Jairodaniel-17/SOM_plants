import streamlit as st
import numpy as np
from PIL import Image
import joblib
from datasets import load_dataset
import plotly.express as px
import plotly.graph_objects as go

# configurar el tamaño de la página de Streamlit
st.set_page_config(layout="wide")
st.sidebar.title("Clasificación de Imágenes con SOM")

st.sidebar.info(
    """
    Resumen: Este es un prototipo de aplicación web que utiliza un Mapa Auto-Organizado (SOM) para clasificar imágenes de plantas.

    Integrantes:
    * **Jairo Daniel Mendoza Torres**
    * **Edson Emanuel Alvarado Prieto**
    * **Daniel Diaz Seminario**
    """
)

st.sidebar.warning(
    "Recuerde que el modelo SOM se entrenó con un conjunto de datos de plantas, por lo que solo se pueden clasificar imágenes de plantas."
)

st.sidebar.write("Imagen de muestra:")
st.sidebar.image("imagen.jpg", caption="Imagen de muestra.", use_column_width=True)

# Cargar el modelo SOM y las etiquetas desde el archivo
som, node_labels = joblib.load("modelo_som_con_labels.pkl")

# Cargar el conjunto de datos original para obtener los nombres de las etiquetas
dataset = load_dataset(
    "jbarat/plant_species", token="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"
)
label_names = dataset["train"].features["label"].names
# Crear un mapeo de índices a nombres de etiquetas y reemplazar guiones bajos por espacios
label_map = {i: label.replace("_", " ") for i, label in enumerate(label_names)}


def preprocess_image(image, size=(64, 64)):
    """
    Preprocesa una imagen cargada desde un archivo en el disco.
    Args:
        image (PIL.Image): La imagen cargada.
        size (tuple): El tamaño al que se redimensionará la imagen.
    """
    image_resized = image.resize(size)
    image_np = np.array(image_resized) / 255.0
    image_flattened = image_np.flatten()
    return image_flattened


# dividir en dos columnas
col1, col2 = st.columns(2)
uploaded_file = None
with col1:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada.", use_column_width=True)

with col2:
    st.title("Resultados de la Clasificación")
    if uploaded_file is not None:
        processed_image = preprocess_image(image)
        winning_node = som.winner(processed_image)
        predicted_label_index = node_labels.get(winning_node, "Desconocido")
        predicted_label = label_map.get(predicted_label_index, "Desconocido")

        st.success(
            f"La nueva imagen se asigna al nodo: {winning_node} con la etiqueta: {predicted_label} (índice: {predicted_label_index})"
        )

        # Visualizar las etiquetas asignadas a los nodos con colores
        label_map_img = np.zeros(
            (som._weights.shape[0], som._weights.shape[1]), dtype=int
        )
        for (x, y), label in node_labels.items():
            label_map_img[x, y] = label

        # Asignar colores únicos a cada etiqueta
        unique_labels = np.unique(list(node_labels.values()))
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
        ]

        # Crear la matriz de anotaciones (índices)
        annotations = []
        for i in range(label_map_img.shape[0]):
            for j in range(label_map_img.shape[1]):
                annotations.append(
                    go.layout.Annotation(
                        text=str(label_map_img[i, j]),
                        x=j,
                        y=i,
                        xref="x1",
                        yref="y1",
                        showarrow=False,
                        font=dict(color="black"),
                    )
                )

        # Crear el mapa de calor con Plotly
        fig = go.Figure(
            data=go.Heatmap(
                z=label_map_img,
                colorscale=[
                    (i / (len(colors) - 1), color) for i, color in enumerate(colors)
                ],
                showscale=False,
                zmin=0,
                zmax=len(unique_labels) - 1,
            )
        )

        # Añadir las anotaciones al gráfico
        fig.update_layout(
            title="Etiquetas asignadas a los nodos del SOM",
            xaxis_title="SOM Width",
            yaxis_title="SOM Height",
            annotations=annotations,
        )

        # Añadir la leyenda
        for i, label in enumerate(unique_labels):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=colors[i]),
                    legendgroup=str(i),
                    showlegend=True,
                    name=f"Índice: {i}, Etiqueta: {label_map[label]}",
                )
            )

        st.plotly_chart(fig)

        # https://huggingface.co/datasets/jbarat/plant_species
        st.write(
            "El conjunto de datos utilizado para entrenar el modelo SOM se puede encontrar en [Hugging Face Datasets](https://huggingface.co/datasets/jbarat/plant_species)."
        )
