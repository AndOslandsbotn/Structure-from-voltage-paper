import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def load_data(filepath):
    mds_emb = np.genfromtxt(os.path.join(filepath, 'mds_emb'), delimiter=',')
    v_emb =  np.genfromtxt(os.path.join(filepath, 'v_emb'), delimiter=',')
    # idx_lm is a csv with rows of different length so we need to load this abit differently
    lm_indices_all_lm = []
    with open(os.path.join(filepath, 'idx_lm'), 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Convert the row to a NumPy array and append it to the list
            array = np.array(row, dtype=int)
            lm_indices_all_lm.append(array)
    return mds_emb, v_emb, lm_indices_all_lm

def plot_domain2D(x, lm_indices_all_lm, radius, title):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot()
    ax.scatter(x[:-1, 0], x[:-1, 2], c=radius)
    for lm_indices in lm_indices_all_lm:
        ax.scatter(x[lm_indices, 0],
                    x[lm_indices, 2],
                    marker='s', c='red')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

def plot_domain3D(x, lm_indices_all_lm, radius, title):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:-1, 0], x[:-1, 1], x[:-1, 2], c=radius)
    for lm_indices in lm_indices_all_lm:
        ax.scatter(x[lm_indices, 0],
                    x[lm_indices, 1],
                    x[lm_indices, 2],
                    marker='s', c='red')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.view_init(-0, 0)

    plt.show()


if __name__ == '__main__':
    m = 3 # Number of landmarks
    filepath = f'Results16may2024smallRadius/SphereTheta90Phi180nlm{m}'
    #filepath = f'Results16may2024largeRadius/SphereTheta90Phi180nlm{m}'
    mds_emb, v_emb, lm_indices_all_lm = load_data(filepath)
    radius = v_emb[0:len(mds_emb)-1, 0]
    plot_domain3D(mds_emb, lm_indices_all_lm, radius, title=f'sphere embedding with {m} landmarks')