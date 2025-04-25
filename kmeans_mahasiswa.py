# ============================
# 1. IMPORT LIBRARY
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D  # Untuk plot 3D
import plotly.graph_objects as go

# ============================
# 2. LOAD DATA
# ============================
print("[INFO] Memuat data...")
file_path = "data_kuesioner_mahasiswa_AI.xlsx"
df = pd.read_excel(file_path)
print("\n5 baris pertama data:")
print(df.head())

# ============================
# 3. PREPROCESSING
# ============================
print("\n[INFO] Pra-pemrosesan data...")

original_columns = [
    'NO',
    'Seberapa sering Anda menggunakan AI',
    'Seberapa sering AI membantu Anda dalam Mencari Referensi Tugas?',
    'Seberapa sering AI membantu Anda dalam Menyusun Kerangka Tugas?',
    'Seberapa sering AI membantu Anda dalam Mengecek grammar/plagiarisme ?',
    'AI meningkatkan kualitas tugas saya',
    'AI membuat saya lebih efisien',
    'Saya merasa ketergantungan pada AI untuk menyelesaikan tugas',
    'Penggunaan AI mengurangi pemahaman mendalam saya terhadap materi',
    'Rata-rata waktu penyelesaian tugas tanpa AI',
    'Rata-rata waktu penyelesaian tugas dengan AI',
    'Rata-rata nilai tugas sebelum menggunakan AI',
    'Rata-rata nilai tugas setelah menggunakan AI'
]

column_mapping = {
    original_columns[1]: 'Frekuensi_AI',
    original_columns[2]: 'AI_Referensi',
    original_columns[3]: 'AI_Kerangka',
    original_columns[4]: 'AI_Grammar',
    original_columns[5]: 'Kualitas_Tugas',
    original_columns[6]: 'Efisiensi',
    original_columns[7]: 'Ketergantungan',
    original_columns[8]: 'Reduksi_Pemahaman',
    original_columns[9]: 'Waktu_Tanpa_AI',
    original_columns[10]: 'Waktu_Dengan_AI',
    original_columns[11]: 'Nilai_Sebelum_AI',
    original_columns[12]: 'Nilai_Setelah_AI'
}

df = df.rename(columns=column_mapping)
features = list(column_mapping.values())
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 4.1. METODE SILHOUETTE
# ============================
print("\n[INFO] Menghitung Silhouette Score...")
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Untuk k={k}, Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='red')
plt.title('Silhouette Score Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid()
plt.show()

# ============================
# 4.2. ELBOW METHOD
# ============================
print("\n[INFO] Menentukan jumlah cluster optimal...")
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='blue', linestyle='--')
plt.title('Elbow Method (Optimal k = 3)', fontsize=14, pad=20)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate('Elbow Point', xy=(3, wcss[2]), xytext=(4, wcss[2]+50),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# ============================
# 5. CLUSTERING DENGAN K-MEANS
# ============================
optimal_k = 3
print(f"\n[INFO] Menjalankan K-Means dengan {optimal_k} cluster...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ============================
# 5.1 TABEL JARAK & CLUSTER MAHASISWA
# ============================
print("\n[INFO] Membuat tabel jarak antar centroid...")

# Hitung jarak antar mahasiswa ke tiap centroid
distances = kmeans.transform(X_scaled)  # shape: (n_samples, n_clusters)
df['Jarak_Cluster_0'] = distances[:, 0]
df['Jarak_Cluster_1'] = distances[:, 1]
df['Jarak_Cluster_2'] = distances[:, 2]

# Label ulang sesuai deskripsi
label_mapping = {
    0: "Penggunaan AI Seimbang (Optimal)",
    1: "Pengguna Minimal",
    2: "Pengguna Intensif (Berisiko Ketergantungan)"
}
df['Label_Cluster'] = df['Cluster'].map(label_mapping)

# Pilih 35 data pertama
tabel_visual = df[['NO', 'Cluster', 'Jarak_Cluster_0', 'Jarak_Cluster_1', 'Jarak_Cluster_2', 'Label_Cluster']].head(35)

print("\n[INFO] Tabel Mahasiswa dan Cluster:")
print(tabel_visual.to_string(index=False))  # Print rapi tanpa index

# (Optional) Simpan ke Excel juga
tabel_visual.to_excel("tabel_cluster_mahasiswa_1_35.xlsx", index=False)


# ============================
# 6. ANALISIS CLUSTER
# ============================
print("\n[INFO] Analisis karakteristik cluster:")
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)

# Visualisasi dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.title('Visualisasi Cluster dengan PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_summary.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap Rata-Rata Fitur per Cluster', pad=20)
plt.xlabel('Cluster')
plt.ylabel('Fitur')
plt.xticks(rotation=0)
plt.show()

# PCA 3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
df['PCA3'] = X_pca_3d[:, 2]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                     c=df['Cluster'], cmap='viridis', s=60)

ax.set_title('Visualisasi 3D PCA Clustering', pad=20)
ax.set_xlabel('PC1 (Intensitas AI)')
ax.set_ylabel('PC2 (Dampak Akademik)')
ax.set_zlabel('PC3')
plt.colorbar(scatter)
plt.show()

# Radar chart
print("\n[INFO] Membuat profil cluster yang lebih informatif...")

def plot_radar_chart(cluster_data, features):
    categories = features
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for cluster, row in cluster_data.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)

    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=8)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Profil Cluster (Radar Chart)', pad=20)
    plt.show()

selected_features = ['Frekuensi_AI', 'Ketergantungan', 'Kualitas_Tugas', 
                    'Nilai_Setelah_AI', 'Waktu_Dengan_AI', 'Reduksi_Pemahaman']
plot_radar_chart(cluster_summary[selected_features], selected_features)

# Plotly Interaktif
fig = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3',
                    color='Cluster', 
                    hover_name='NO',
                    hover_data=features,
                    title='Visualisasi 3D Interaktif Cluster Mahasiswa',
                    color_continuous_scale='viridis',
                    width=1000, height=800)
fig.update_layout(scene=dict(
                    xaxis_title='PC1 - Intensitas AI',
                    yaxis_title='PC2 - Dampak Akademik',
                    zaxis_title='PC3'),
                  margin=dict(l=0, r=0, b=0, t=30))
fig.show()

# ============================
# 11. LABEL KATEGORI CLUSTER & PERSENTASE
# ============================

# Ubah nama kolom kategori cluster agar lebih deskriptif
df['Label_Cluster'] = df['Cluster'].map(label_mapping)

# Tambahkan kolom jarak ke masing-masing cluster
centroids = kmeans.cluster_centers_
jarak_clusters = np.linalg.norm(X_scaled[:, np.newaxis] - centroids, axis=2)
df['Jarak_Cluster_0'] = jarak_clusters[:, 0]
df['Jarak_Cluster_1'] = jarak_clusters[:, 1]
df['Jarak_Cluster_2'] = jarak_clusters[:, 2]

# Ambil 35 data pertama
tabel_visual = df[['NO', 'Cluster', 'Jarak_Cluster_0', 'Jarak_Cluster_1', 'Jarak_Cluster_2', 'Label_Cluster']].head(35)

# Buat tabel interaktif
fig = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>No</b>", "<b>Cluster</b>", "<b>Jarak ke Cluster 0</b>", "<b>Jarak ke Cluster 1</b>", "<b>Jarak ke Cluster 2</b>", "<b>Deskripsi Cluster</b>"],
        fill_color='paleturquoise',
        align='center',
        font=dict(size=12, color='black')
    ),
    cells=dict(
        values=[
            tabel_visual['NO'],
            tabel_visual['Cluster'],
            tabel_visual['Jarak_Cluster_0'].round(3),
            tabel_visual['Jarak_Cluster_1'].round(3),
            tabel_visual['Jarak_Cluster_2'].round(3),
            tabel_visual['Label_Cluster']
        ],
        fill_color='lavender',
        align='center',
        font=dict(size=11),
        height=30
    )
)])

fig.update_layout(
    title="Tabel Interaktif: Mahasiswa dan Hasil Clustering",
    width=1000,
    height=800
)
fig.show()

print("\n[INFO] Menambahkan label deskriptif untuk masing-masing cluster...")
label_mapping = {
    0: "Penggunaan AI Seimbang (Optimal)",
    1: "Pengguna Minimal",
    2: "Pengguna Intensif (Berisiko Ketergantungan)"
}
df['Kategori_Cluster'] = df['Cluster'].map(label_mapping)
cluster_counts = df['Kategori_Cluster'].value_counts(normalize=True) * 100
print("\nDistribusi Mahasiswa per Kategori Cluster (%):")
print(cluster_counts.round(2))

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette("Set2"))
plt.title("Distribusi Pengguna AI Berdasarkan Kategori Cluster", pad=20)
plt.show()

# Barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set2")
plt.title("Persentase Mahasiswa per Kategori Cluster", pad=20)
plt.ylabel("Persentase (%)")
plt.xticks(rotation=15)
plt.ylim(0, 100)
for index, value in enumerate(cluster_counts.values):
    plt.text(index, value + 1, f"{value:.1f}%", ha='center', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ============================
# 12. DISTRIBUSI CLUSTER MAHASISWA
# ============================
print("\n[INFO] Distribusi Mahasiswa per Cluster:")

# Hitung jumlah & persentase
cluster_counts = df['Cluster'].value_counts().sort_index()
cluster_percent = (cluster_counts / cluster_counts.sum() * 100).round(2)

# Label kategori
cluster_labels = {
    0: "Penggunaan AI Seimbang (Ideal)",
    1: "Pengguna Minimal",
    2: "Pengguna Intensif"
}

# Gabungkan ke dalam DataFrame
cluster_distribution = pd.DataFrame({
    'Jumlah Mahasiswa': cluster_counts,
    'Persentase (%)': cluster_percent,
    'Deskripsi': cluster_counts.index.map(cluster_labels)
})

print(cluster_distribution)


# ============================
# 13. SIMPAN HASIL ANALISIS
# ============================
output_file = "hasil_clustering_mahasiswa_AI.xlsx"
df.to_excel(output_file, index=False)
cluster_stats = df.groupby('Cluster')[features].agg(['mean', 'std', 'count'])
with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
    cluster_stats.to_excel(writer, sheet_name='Cluster_Statistics')
    pd.DataFrame(pca.components_, columns=features, 
                index=['PC1', 'PC2']).to_excel(writer, sheet_name='PCA_Components')

print("\n[INFO] Analisis tambahan:")
print(f"1. Silhouette Score terbaik: {max(silhouette_scores):.4f} pada k={optimal_k}")
print("2. Hasil lengkap disimpan dalam sheet tambahan di file Excel")
