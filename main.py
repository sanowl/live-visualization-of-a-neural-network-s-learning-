import tensorflow as tf
from tensorflow import keras
from datasets import load_dataset
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx

# Load the MNIST dataset from Hugging Face
dataset = load_dataset("mnist", trust_remote_code=True)
train_data = dataset["train"]
test_data = dataset["test"]

# Preprocess the data
def preprocess_data(data):
    images = np.array([np.array(img) for img in data["image"]])
    labels = np.array(data["label"])
    images = images.reshape((-1, 28, 28, 1)) / 255.0
    return images, labels

train_images, train_labels = preprocess_data(train_data)
test_images, test_labels = preprocess_data(test_data)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Create a graph representation of the neural network
def create_network_graph(model):
    graph = nx.DiGraph()
    for i, layer in enumerate(model.layers):
        layer_name = f"{i}_{layer.name}"
        graph.add_node(layer_name)
        if i > 0:
            prev_layer_name = f"{i-1}_{model.layers[i-1].name}"
            graph.add_edge(prev_layer_name, layer_name)
    return graph

network_graph = create_network_graph(model)

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Neural Network Learning Visualization"),
    dcc.Graph(id="loss-graph"),
    dcc.Graph(id="accuracy-graph"),
    dcc.Graph(id="network-graph"),
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0)
])

# Callback to update the graphs
@app.callback(
    [Output("loss-graph", "figure"),
     Output("accuracy-graph", "figure"),
     Output("network-graph", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_graphs(n):
    # Train the model for one epoch
    history = model.fit(train_dataset, epochs=1, verbose=0)

    # Get the loss and accuracy for the current epoch
    loss = history.history["loss"][0]
    accuracy = history.history["accuracy"][0]

    # Create the loss graph
    loss_graph = go.Figure(
        data=[go.Scatter(x=list(range(1, n + 2)), y=history.history["loss"], mode="lines")],
        layout=go.Layout(
            title="Training Loss",
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Loss")
        )
    )

    # Create the accuracy graph
    accuracy_graph = go.Figure(
        data=[go.Scatter(x=list(range(1, n + 2)), y=history.history["accuracy"], mode="lines")],
        layout=go.Layout(
            title="Training Accuracy",
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Accuracy")
        )
    )

    # Create the network graph
    pos = nx.spring_layout(network_graph)
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            line=dict(width=2)
        )
    )
    for edge in network_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += tuple([x0, x1, None])
        edge_trace["y"] += tuple([y0, y1, None])
    for node in network_graph.nodes():
        x, y = pos[node]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])
        node_trace["text"] += tuple([node])

    network_graph_figure = go.Figure(data=[edge_trace, node_trace],
                                     layout=go.Layout(
                                         title="Neural Network Graph",
                                         showlegend=False,
                                         hovermode="closest",
                                         margin=dict(b=20, l=5, r=5, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                     )

    return loss_graph, accuracy_graph, network_graph_figure

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)