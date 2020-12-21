import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("mushrooms.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:,9] = labelencoder_X.fit_transform(X[:,9])
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,11] = labelencoder_X.fit_transform(X[:,11])
X[:,12] = labelencoder_X.fit_transform(X[:,12])
X[:,13] = labelencoder_X.fit_transform(X[:,13])
X[:,14] = labelencoder_X.fit_transform(X[:,14])
X[:,15] = labelencoder_X.fit_transform(X[:,15])
X[:,16] = labelencoder_X.fit_transform(X[:,16])
X[:,17] = labelencoder_X.fit_transform(X[:,17])
X[:,18] = labelencoder_X.fit_transform(X[:,18])
X[:,19] = labelencoder_X.fit_transform(X[:,19])
X[:,20] = labelencoder_X.fit_transform(X[:,20])
X[:,21] = labelencoder_X.fit_transform(X[:,21])

features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

X = df[features]
y = df['class']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img = pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
