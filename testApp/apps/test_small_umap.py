import umap
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd

sns.set(context="paper", style="white")

iris = load_iris()
irisDF = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
irisDF['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Set random state
np.random.seed(2105)

reducer = umap.UMAP(random_state = 2105, transform_seed = 2105)
embedding = reducer.fit_transform(iris.data)
embedding.shape
# embDF = pd.DataFrame(embedding, index=irisDF['species'].values.tolist(), columns=['UM1','UM2'])
embDF = pd.DataFrame(embedding, columns=['UM1','UM2'])

# Define a nice colour map for gene expression
colors2     = plt.cm.Reds(np.linspace(0, 1, 128))
colors3     = plt.cm.Greys_r(np.linspace(0.7,0.8,20))
colorsComb  = np.vstack([colors3, colors2])
expColorMap = LinearSegmentedColormap.from_list('ExpressionColorMap', colorsComb)


# plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in iris.target])
plt.scatter(embedding[:, 0], embedding[:, 1], c=expColorMap)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Iris dataset', fontsize=24);

plt.show()

# Using plotly.go
import plotly.graph_objs as go
fig = go.Figure()
fvals = irisDF['sepal length (cm)']

markerDict = dict(color=fvals,
                        colorscale='Viridis', opacity=0.9,
                        showscale=True,
                        colorbar=dict(thickness=20,
                                        title="Test",
                                        titleside='right'))

fig.add_trace(go.Scatter(x=embedding[:, 0], y=embedding[:, 1], mode='markers',
                        hoverinfo='text', text='sepal length (cm)', marker = markerDict))

fig.show()
