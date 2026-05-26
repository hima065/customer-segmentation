import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st  # 1. Import Streamlit

# Add a web title and description
st.title("Customer Segmentation Dashboard")
st.write("This page displays the K-Means clustering results for mall customers.")

# Load data
customer_data = pd.read_csv("Mall_Customers.csv")
X = customer_data.iloc[:, [3, 4]].values

# --- PLOT 1: THE ELBOW METHOD ---
st.subheader("1. The Elbow Method")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Create the figure object for Matplotlib
fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss)
ax1.set_title('The Elbow Method')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('WCSS')

st.pyplot(fig1)  # 2. Tell Streamlit to display this plot on the page


# --- PLOT 2: THE CLUSTERS ---
st.subheader("2. Customer Groups Clusters")

# Train final model (assuming 5 clusters based on your elbow plot)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

fig2, ax2 = plt.subplots()
# Plotting the 5 clusters (adjust colors/labels based on your exact script)
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    ax2.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')

ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
ax2.set_title('Customer Groups')
ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score (1-100)')
ax2.legend()

st.pyplot(fig2)  # 3. Tell Streamlit to display the second plot