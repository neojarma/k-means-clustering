from flask_cors import CORS
from flask import Flask, request, send_from_directory, render_template, jsonify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('SVG')


app = Flask(__name__)
CORS(app)


dataset = []
dataset_labels = []
columns = []
rows = None
cluster = None
max_iteration = None
kmeans = None


@app.route('/')
def index():
    return render_template('index.html')


class ClusteringResult():
    def __init__(self) -> None:
        self.iteration = 0
        self.centroids = []
        self.euclideans = []
        self.clusters = []

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'centroids': [c.tolist() for c in self.centroids],
            'euclideans': self.euclideans,
            'clusters': self.clusters
        }


class KMeans():

    def __init__(self, data, k, labels):
        self.data = data
        self.k = k
        self.labels = labels
        self.assignment = [-1 for _ in range(len(data))]
        self.results = []

    def _is_unassigned(self, i):
        return self.assignment[i] == -1

    def _unassign_all(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def _is_centers_diff(self, c1, c2):
        for i in range(self.k):
            if self.dist(c1[i], c2[i]) != 0:
                return True
        return False

    def dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def kmeans_plusplus(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        centroids = self.data[np.random.choice(range(len(self.data)), size=1)]
        for _ in range(1, self.k):
            min_sq_dist = [min([self.dist(c, x) ** 2 for c in centroids])
                           for x in self.data]
            prob = min_sq_dist / sum(min_sq_dist)
            centroids = np.append(centroids, self.data[np.random.choice(
                range(len(self.data)), size=1, p=prob)], axis=0)
        return centroids

    def assign(self, centers):
        temp = ClusteringResult()
        temp.iteration = self.iteration + 1
        temp.centroids = centers
        for i in range(len(self.data)):
            min_dist = float('inf')
            eucTemp = []
            for j in range(self.k):
                d = self.dist(self.data[i], centers[j])
                eucTemp.append(d)
                if d < min_dist:
                    min_dist = d
                    self.assignment[i] = j
            eucTemp.insert(0, self.labels[i])
            temp.euclideans.append(eucTemp)

        for i in range(len(self.data)):
            temp.clusters.append([self.labels[i], self.assignment[i]+1])

        self.results.append(temp)

    def compute_centers(self):
        centers = []
        for j in range(self.k):
            cluster = np.array([self.data[k] for k in filter(lambda x: x >= 0,
                                                             [i if self.assignment[i] == j else -1 for i in range(len(self.data))])])
            centers.append(np.mean(cluster, axis=0))
        return np.array(centers)

    def lloyds(self, max_iter=100, seed=None):
        self.iteration = 0
        centers = self.kmeans_plusplus(seed=seed)
        self.assign(centers)
        new_centers = self.compute_centers()
        while self._is_centers_diff(centers, new_centers) and self.iteration < max_iter:
            self._unassign_all()
            self.iteration += 1
            centers = new_centers
            self.assign(centers)
            new_centers = self.compute_centers()

    def printResults(self):
        for x in self.results:
            print(x.iteration)
            print(x.centroids)
            print(x.euclideans)
            print(x.clusters)

    def plotElbow(self, max_k):
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(self.data, k, self.labels)
            kmeans.lloyds()
            inertia = 0
            for result in kmeans.results:
                inertia += sum(min(euc[1:]) ** 2 for euc in result.euclideans)
            inertias.append(inertia)

        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.savefig('images/elbow.png')
        plt.close()

    def plot_clusters(self):
        # Get the last clustering result
        last_result = self.results[-1]
        centroids = last_result.centroids
        assignments = np.array(self.assignment)

        # Assign colors to clusters
        colors = plt.cm.get_cmap('tab10', self.k)

        if self.data.shape[1] >= 3:
            # 3D plot for the first three features
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(self.k):
                cluster_data = self.data[assignments == i]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], s=30, c=[
                           colors(i)], label=f'Cluster {i+1}')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       s=200, c='black', marker='X', label='Centroids')
            ax.set_title('Cluster Assignments and Centroids')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            plt.legend()
            plt.savefig('images/cluster.png')
            plt.close()
        else:
            # 2D plot for the first two features
            for i in range(self.k):
                cluster_data = self.data[assignments == i]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=30, c=[
                            colors(i)], label=f'Cluster {i+1}')
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        s=200, c='black', marker='X', label='Centroids')
            plt.title('Cluster Assignments and Centroids')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.savefig('images/cluster.png')
            plt.close()


@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset, dataset_labels, columns, rows, kmeans

    file = request.files['file']
    rd = pd.read_excel(file)

    dataset = rd.iloc[:, 1:].values
    dataset_labels = rd.iloc[:, 0].values
    columns = rd.columns
    rows = len(dataset)

    kmeans = KMeans(dataset, 4, dataset_labels)

    return jsonify({"dataset": dataset.tolist()[:5], "labels": dataset_labels.tolist()[:5], "columns": columns.to_list(), "rows": rows}), 200


@app.route('/cluster', methods=['POST'])
def cluster_config():
    global kmeans, dataset, dataset_labels, max_iteration, cluster
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    data = request.json

    max_iteration = data['iteration']
    cluster = data['cluster']

    return jsonify({'message': 'Success set cluster'})


@app.route('/k-means', methods=['GET'])
def kmeans_process():
    global kmeans, dataset, dataset_labels, cluster
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    kmeans = KMeans(dataset, cluster, dataset_labels)
    kmeans.lloyds(max_iteration, 42)

    return jsonify({'result': [result.to_dict() for result in kmeans.results]})


@app.route('/elbow', methods=['POST'])
def plot_elbow():
    global kmeans, dataset
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    data = request.json
    kmeans.plotElbow(data['max_cluster'])
    return jsonify({"path": 'http://' + request.host + '/images/elbow.png'})


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/cluster', methods=['GET'])
def plot_clusters():
    global kmeans, dataset, dataset_labels, cluster
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    if cluster == None:
        return jsonify({'message': 'Please complete the K-Means processing first.'}), 400

    kmeans = KMeans(dataset, cluster, dataset_labels)
    kmeans.lloyds(max_iteration, 42)
    iterations = [result.to_dict() for result in kmeans.results][-1]

    kmeans.plot_clusters()
    clusterPath = 'http://' + request.host + '/images/cluster.png'

    return jsonify({'iterations': iterations, 'clusterPath': clusterPath})


if __name__ == '__main__':
    app.run(debug=True)
