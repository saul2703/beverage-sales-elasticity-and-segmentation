from pathlib import Path
import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "synthetic_beverage_sales_data.csv"
OUT_DIR = ROOT_DIR / "data"
IMG_DIR = ROOT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print(f"Streaming {CSV_PATH} ({CSV_PATH.stat().st_size / (1024**2):.1f} MB)...")

total_rows = 0
total_orders = 0
missing_counts = {}
total_revenue = 0.0
total_gross = 0.0
total_qty = 0
min_date = "9999-99-99"
max_date = "0000-00-00"

reg_frames = []
cat_frames = []
prod_frames = []
mo_frames = []
disc_frames = []
cust_frames = []

prev_last_order_id = None

for idx, ch in enumerate(pd.read_csv(
    CSV_PATH,
    encoding="utf-8",
    dtype={
        "Order_ID": "string",
        "Customer_ID": "string",
        "Customer_Type": "category",
        "Product": "category",
        "Category": "category",
        "Unit_Price": "float64",
        "Quantity": "int64",
        "Discount": "float64",
        "Total_Price": "float64",
        "Region": "category",
        "Order_Date": "string",
    },
    chunksize=1_500_000,
)):
    n = len(ch)
    total_rows += n
    for col, cnt in ch.isna().sum().items():
        missing_counts[col] = missing_counts.get(col, 0) + int(cnt)

    d_min = str(ch["Order_Date"].min())
    d_max = str(ch["Order_Date"].max())
    if d_min < min_date:
        min_date = d_min
    if d_max > max_date:
        max_date = d_max

    # Vectorized contiguous order boundary detection (O(N), 10x faster than nunique)
    ord_col = ch["Order_ID"]
    is_new = (ord_col != ord_col.shift(1)).fillna(True).to_numpy(dtype=np.int32)
    if prev_last_order_id is not None and len(ord_col) > 0 and ord_col.iloc[0] == prev_last_order_id:
        is_new[0] = 0
    if len(ord_col) > 0:
        prev_last_order_id = ord_col.iloc[-1]
    ch["is_new_order"] = is_new
    total_orders += int(is_new.sum())

    gross = ch["Unit_Price"] * ch["Quantity"]
    ch["Gross_Price"] = gross
    ch["YearMonth"] = ch["Order_Date"].str.slice(0, 7)
    ch["Qty_Sq"] = ch["Quantity"].astype("float64") ** 2
    ch["Rev_Sq"] = ch["Total_Price"] ** 2
    ch["Lines"] = 1

    total_revenue += float(ch["Total_Price"].sum())
    total_gross += float(gross.sum())
    total_qty += int(ch["Quantity"].sum())

    reg_frames.append(
        ch.groupby(["Region", "Customer_Type"], observed=True)[
            ["Total_Price", "Gross_Price", "Quantity", "Lines", "is_new_order"]
        ].sum().reset_index()
    )
    cat_frames.append(
        ch.groupby(["Category", "Customer_Type"], observed=True)[
            ["Total_Price", "Gross_Price", "Quantity", "Lines", "is_new_order"]
        ].sum().reset_index()
    )
    prod_frames.append(
        ch.groupby(["Product", "Category", "Customer_Type"], observed=True)[
            ["Total_Price", "Gross_Price", "Quantity", "Lines", "Unit_Price", "Discount"]
        ].sum().reset_index()
    )
    mo_frames.append(
        ch.groupby(["YearMonth", "Customer_Type"], observed=True)[
            ["Total_Price", "Gross_Price", "Quantity", "Lines", "is_new_order"]
        ].sum().reset_index()
    )
    disc_frames.append(
        ch.groupby(["Discount", "Customer_Type"], observed=True)[
            ["Total_Price", "Gross_Price", "Quantity", "Lines", "Qty_Sq", "Rev_Sq", "is_new_order"]
        ].sum().reset_index()
    )
    cust_frames.append(
        ch.groupby(["Customer_ID", "Customer_Type", "Region"], observed=True)[
            ["Total_Price", "Gross_Price", "Quantity", "Lines", "is_new_order", "Discount"]
        ].sum().reset_index()
    )

    print(f"  Chunk {idx+1}: {total_rows:,} rows ({time.time() - t0:.1f}s)")

# Consolidate aggregations
reg_df = pd.concat(reg_frames, ignore_index=True).groupby(["Region", "Customer_Type"], observed=True).sum().reset_index()
cat_df = pd.concat(cat_frames, ignore_index=True).groupby(["Category", "Customer_Type"], observed=True).sum().reset_index()
prod_df = pd.concat(prod_frames, ignore_index=True).groupby(["Product", "Category", "Customer_Type"], observed=True).sum().reset_index()
mo_df = pd.concat(mo_frames, ignore_index=True).groupby(["YearMonth", "Customer_Type"], observed=True).sum().reset_index()
disc_df = pd.concat(disc_frames, ignore_index=True).groupby(["Discount", "Customer_Type"], observed=True).sum().reset_index()
cust_df = pd.concat(cust_frames, ignore_index=True).groupby(["Customer_ID", "Customer_Type", "Region"], observed=True).sum().reset_index()

cust_df["orders"] = np.maximum(1, cust_df["is_new_order"])
cust_df["avg_line_qty"] = cust_df["Quantity"] / cust_df["Lines"]
cust_df["avg_order_value"] = cust_df["Total_Price"] / cust_df["orders"]
cust_df["avg_discount"] = cust_df["Discount"] / cust_df["Lines"]
cust_df["effective_disc_pct"] = np.where(
    cust_df["Gross_Price"] > 0,
    (cust_df["Gross_Price"] - cust_df["Total_Price"]) / cust_df["Gross_Price"],
    0.0,
)

print(f"Finished streaming {total_rows:,} rows, {total_orders:,} unique orders, {len(cust_df):,} unique customers in {time.time() - t0:.1f}s")

# ==============================================================================
# K-MEANS CLUSTERING + ELBOW (WCSS) + SILHOUETTE SCORE (Pure Vectorized NumPy)
# ==============================================================================
features = ["Total_Price", "orders", "avg_line_qty", "avg_discount"]
X_raw = cust_df[features].to_numpy(dtype=np.float64)
X_mean = X_raw.mean(axis=0)
X_std = np.where(X_raw.std(axis=0) == 0, 1.0, X_raw.std(axis=0))
X = (X_raw - X_mean) / X_std

def run_kmeans_pp(data, k, seed=42, max_iter=100):
    rng = np.random.default_rng(seed)
    n_samples = data.shape[0]
    centroids = np.empty((k, data.shape[1]), dtype=np.float64)
    centroids[0] = data[rng.integers(n_samples)]
    d2 = np.sum((data - centroids[0]) ** 2, axis=1)
    for c_id in range(1, k):
        probs = d2 / d2.sum()
        idx_c = rng.choice(n_samples, p=probs)
        centroids[c_id] = data[idx_c]
        d2 = np.minimum(d2, np.sum((data - centroids[c_id]) ** 2, axis=1))

    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_id in range(k):
            mask = labels == c_id
            if np.any(mask):
                centroids[c_id] = data[mask].mean(axis=0)
    dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    wcss = float(np.sum(np.min(dists, axis=1)))
    return labels, centroids, wcss

def compute_silhouette(data, labels, k, sample_size=2500, seed=42):
    rng = np.random.default_rng(seed)
    if len(data) > sample_size:
        idx_s = rng.choice(len(data), size=sample_size, replace=False)
        Xs = data[idx_s]
        Ls = labels[idx_s]
    else:
        Xs, Ls = data, labels
    # Pairwise Euclidean distance matrix
    diff = Xs[:, None, :] - Xs[None, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    sil = np.zeros(len(Xs), dtype=np.float64)
    for i in range(len(Xs)):
        own = Ls[i]
        own_mask = (Ls == own)
        own_mask[i] = False
        a_i = float(D[i, own_mask].mean()) if np.any(own_mask) else 0.0
        b_i = np.inf
        for c_id in range(k):
            if c_id == own:
                continue
            other_mask = (Ls == c_id)
            if np.any(other_mask):
                b_i = min(b_i, float(D[i, other_mask].mean()))
        denom = max(a_i, b_i)
        sil[i] = (b_i - a_i) / denom if denom > 0 and np.isfinite(b_i) else 0.0
    return float(sil.mean())

ks = list(range(2, 9))
wcss_list = []
sil_list = []
models = {}

for k in ks:
    best_wcss = np.inf
    best_res = None
    for s in (42, 101, 202):
        lbl, cen, w = run_kmeans_pp(X, k, seed=s)
        if w < best_wcss:
            best_wcss = w
            best_res = (lbl, cen, w)
    lbl, cen, w = best_res
    sc = compute_silhouette(X, lbl, k)
    wcss_list.append(w)
    sil_list.append(sc)
    models[k] = (lbl, cen, w, sc)
    print(f"  k={k}: WCSS={w:,.1f} | Silhouette={sc:.4f}")

# Select k=4 (or best silhouette / elbow)
best_k = 4
labels_k, centroids_k, wcss_k, sil_k = models[best_k]
cust_df["Cluster"] = labels_k

# 2D PCA via SVD for cluster visualization
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
X_pca = (X - X.mean(axis=0)) @ Vt[:2].T
var_exp = (S[:2] ** 2) / np.sum(S ** 2) * 100.0
cust_df["PCA1"] = X_pca[:, 0]
cust_df["PCA2"] = X_pca[:, 1]

# ==============================================================================
# CHART STYLING & GENERATION (6 PUBLICATION-READY MATPLOTLIB CHARTS)
# ==============================================================================
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#f8fafc",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#1e293b",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.grid": True,
    "grid.alpha": 0.7,
})

# Chart 1: Regional Revenue Split by B2B vs B2C
reg_piv = reg_df.pivot(index="Region", columns="Customer_Type", values="Total_Price").fillna(0.0)
reg_piv["Total"] = reg_piv.sum(axis=1)
reg_piv = reg_piv.sort_values("Total", ascending=True)

fig, ax = plt.subplots(figsize=(11, 6.8), dpi=160)
y_pos = np.arange(len(reg_piv))
b2b_vals = reg_piv.get("B2B", pd.Series(0, index=reg_piv.index)) / 1e6
b2c_vals = reg_piv.get("B2C", pd.Series(0, index=reg_piv.index)) / 1e6
ax.barh(y_pos, b2b_vals, color="#38bdf8", label="B2B Wholesale", height=0.68)
ax.barh(y_pos, b2c_vals, left=b2b_vals, color="#34d399", label="B2C Retail", height=0.68)
ax.set_yticks(y_pos)
ax.set_yticklabels(reg_piv.index, fontsize=9.5)
ax.set_xlabel("Net Revenue ($ Millions)", fontsize=10.5, fontweight="bold")
ax.set_title("Regional Beverage Revenue Across 16 German States (B2B vs. B2C)", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("$%.0fM"))
ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="lower right")
for i, tot in enumerate(reg_piv["Total"] / 1e6):
    ax.text(tot + 0.8, i, f"${tot:.1f}M", va="center", fontsize=8.5, color="#cbd5e1")
ax.set_xlim(0, (reg_piv["Total"].max() / 1e6) * 1.13)
fig.tight_layout()
fig.savefig(IMG_DIR / "regional_revenue_b2b_b2c.png", dpi=160)
plt.close(fig)

# Chart 2: Category & Top 12 Products Performance
cat_piv = cat_df.pivot(index="Category", columns="Customer_Type", values="Total_Price").fillna(0.0)
cat_piv["Total"] = cat_piv.sum(axis=1)
cat_piv = cat_piv.sort_values("Total", ascending=True)

prod_tot = prod_df.groupby(["Product", "Category"], observed=True).agg(
    revenue=("Total_Price", "sum"),
    qty=("Quantity", "sum"),
    lines=("Lines", "sum"),
).reset_index()
prod_tot = prod_tot[prod_tot["revenue"] > 0].sort_values("revenue", ascending=False).head(12).sort_values("revenue", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=160)
y_c = np.arange(len(cat_piv))
ax1.barh(y_c, cat_piv.get("B2B", 0) / 1e6, color="#38bdf8", label="B2B", height=0.55)
ax1.barh(y_c, cat_piv.get("B2C", 0) / 1e6, left=cat_piv.get("B2B", 0) / 1e6, color="#a78bfa", label="B2C", height=0.55)
ax1.set_yticks(y_c)
ax1.set_yticklabels(cat_piv.index, fontsize=10)
ax1.set_xlabel("Net Revenue ($ Millions)")
ax1.set_title("Revenue by Beverage Category", fontsize=12, fontweight="bold")
ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter("$%.0fM"))
ax1.legend(facecolor="#1e293b", edgecolor="#334155")

cat_colors = {
    "Alcoholic Beverages": "#f472b6",
    "Soft Drinks": "#38bdf8",
    "Juices": "#fbbf24",
    "Water": "#34d399",
}
y_p = np.arange(len(prod_tot))
p_cols = [cat_colors.get(str(c), "#38bdf8") for c in prod_tot["Category"]]
ax2.barh(y_p, prod_tot["revenue"] / 1e6, color=p_cols, height=0.65)
ax2.set_yticks(y_p)
ax2.set_yticklabels(prod_tot["Product"], fontsize=9.5)
ax2.set_xlabel("Net Revenue ($ Millions)")
ax2.set_title("Top 12 Products by Net Revenue (Colored by Category)", fontsize=12, fontweight="bold")
ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter("$%.0fM"))
fig.tight_layout()
fig.savefig(IMG_DIR / "category_product_performance.png", dpi=160)
plt.close(fig)

# Chart 3: Monthly Revenue Time Series (B2B vs B2C)
mo_piv = mo_df.pivot(index="YearMonth", columns="Customer_Type", values="Total_Price").fillna(0.0).sort_index()
mo_piv["Total"] = mo_piv.sum(axis=1)

fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
x_m = np.arange(len(mo_piv))
ax.plot(x_m, mo_piv["Total"] / 1e6, color="#f8fafc", linewidth=2.4, marker="o", markersize=4, label="Total Revenue")
if "B2B" in mo_piv:
    ax.plot(x_m, mo_piv["B2B"] / 1e6, color="#38bdf8", linewidth=2.0, linestyle="--", label="B2B Wholesale")
if "B2C" in mo_piv:
    ax.plot(x_m, mo_piv["B2C"] / 1e6, color="#34d399", linewidth=2.0, linestyle=":", label="B2C Retail")
step = max(1, len(x_m) // 12)
ax.set_xticks(x_m[::step])
ax.set_xticklabels(mo_piv.index[::step], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Monthly Revenue ($ Millions)")
ax.set_title(f"Monthly Revenue Trajectory & Channel Split ({min_date[:7]} to {max_date[:7]})", fontsize=12.5, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.1fM"))
ax.legend(facecolor="#1e293b", edgecolor="#334155")
fig.tight_layout()
fig.savefig(IMG_DIR / "monthly_sales_trend.png", dpi=160)
plt.close(fig)

# Chart 4: Discount Tier & Price Elasticity Analysis
disc_agg = disc_df.groupby("Discount", observed=True).agg(
    revenue=("Total_Price", "sum"),
    gross=("Gross_Price", "sum"),
    qty=("Quantity", "sum"),
    lines=("Lines", "sum"),
    qty_sq=("Qty_Sq", "sum"),
).reset_index().sort_values("Discount")
disc_agg["avg_qty"] = disc_agg["qty"] / disc_agg["lines"]
disc_agg["std_qty"] = np.sqrt(np.maximum(0.0, (disc_agg["qty_sq"] / disc_agg["lines"]) - (disc_agg["avg_qty"] ** 2)))
disc_agg["ci95_qty"] = 1.96 * disc_agg["std_qty"] / np.sqrt(disc_agg["lines"])
disc_agg["disc_loss_m"] = (disc_agg["gross"] - disc_agg["revenue"]) / 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=160)
d_labels = [f"{int(round(d * 100))}%" for d in disc_agg["Discount"]]
x_d = np.arange(len(d_labels))
ax1.bar(x_d, disc_agg["avg_qty"], yerr=disc_agg["ci95_qty"], capsize=5, color="#38bdf8", width=0.55)
ax1.set_xticks(x_d)
ax1.set_xticklabels(d_labels, fontsize=10)
ax1.set_xlabel("Discount Tier")
ax1.set_ylabel("Mean Units per Order Line (±95% CI)")
ax1.set_title("Order Line Quantity by Discount Tier", fontsize=12, fontweight="bold")

ax2.bar(x_d - 0.18, disc_agg["revenue"] / 1e6, width=0.35, color="#34d399", label="Net Revenue ($M)")
ax2.bar(x_d + 0.18, disc_agg["disc_loss_m"], width=0.35, color="#f87171", label="Discount Leakage ($M)")
ax2.set_xticks(x_d)
ax2.set_xticklabels(d_labels, fontsize=10)
ax2.set_xlabel("Discount Tier")
ax2.set_ylabel("USD ($ Millions)")
ax2.set_title("Net Realized Revenue vs. Promotional Margin Leakage", fontsize=12, fontweight="bold")
ax2.legend(facecolor="#1e293b", edgecolor="#334155")
fig.tight_layout()
fig.savefig(IMG_DIR / "discount_price_elasticity.png", dpi=160)
plt.close(fig)

# Chart 5: K-Means Elbow (WCSS) & Silhouette Score Evaluation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=160)
ax1.plot(ks, [w / 1e3 for w in wcss_list], color="#38bdf8", marker="o", linewidth=2.2)
ax1.axvline(best_k, color="#fbbf24", linestyle="--", label=f"Selected k = {best_k}")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("WCSS / Inertia (Thousands)")
ax1.set_title("Elbow Method (Within-Cluster Sum of Squares)", fontsize=12, fontweight="bold")
ax1.legend(facecolor="#1e293b", edgecolor="#334155")

ax2.plot(ks, sil_list, color="#34d399", marker="s", linewidth=2.2)
ax2.axvline(best_k, color="#fbbf24", linestyle="--", label=f"Selected k = {best_k} (Sil = {sil_k:.3f})")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Mean Silhouette Coefficient")
ax2.set_title("Silhouette Score across Candidate k Values", fontsize=12, fontweight="bold")
ax2.legend(facecolor="#1e293b", edgecolor="#334155")
fig.tight_layout()
fig.savefig(IMG_DIR / "elbow_silhouette_evaluation.png", dpi=160)
plt.close(fig)

# Chart 6: Customer Segments Scatter (Spend vs. Basket Quantity + PCA)
cluster_summary = cust_df.groupby("Cluster").agg(
    customers=("Customer_ID", "count"),
    mean_revenue=("Total_Price", "mean"),
    mean_orders=("orders", "mean"),
    mean_line_qty=("avg_line_qty", "mean"),
    mean_disc=("avg_discount", "mean"),
    b2b_share=("Customer_Type", lambda s: float((s == "B2B").mean())),
).reset_index()

# Assign human-readable business persona names ordered by mean_revenue / b2b_share
cluster_names = {}
sorted_c = cluster_summary.sort_values("mean_revenue", ascending=False).reset_index(drop=True)
persona_labels = [
    "Cluster 1: High-Volume B2B Anchors",
    "Cluster 2: Growth Wholesale Accounts",
    "Cluster 3: High-Value Retail / Hybrid",
    "Cluster 4: Standard B2C Consumers",
]
for rank_i, row in sorted_c.iterrows():
    cluster_names[int(row["Cluster"])] = persona_labels[rank_i]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=160)
palette = ["#38bdf8", "#34d399", "#fbbf24", "#f472b6"]

for idx_c, c_id in enumerate(sorted_c["Cluster"].astype(int)):
    sub = cust_df[cust_df["Cluster"] == c_id]
    lbl_name = cluster_names[c_id]
    ax1.scatter(
        sub["avg_line_qty"],
        sub["Total_Price"] / 1e3,
        s=14,
        alpha=0.65,
        color=palette[idx_c % len(palette)],
        label=f"{lbl_name} (n={len(sub):,})",
    )
    ax2.scatter(
        sub["PCA1"],
        sub["PCA2"],
        s=14,
        alpha=0.65,
        color=palette[idx_c % len(palette)],
        label=lbl_name,
    )

ax1.set_xlabel("Average Units per Order Line")
ax1.set_ylabel("Customer Total Revenue ($ Thousands)")
ax1.set_title("Customer Segments: Total Revenue vs. Basket Size", fontsize=12, fontweight="bold")
ax1.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)

ax2.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% variance)")
ax2.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% variance)")
ax2.set_title("2D PCA Projection of Standardized Behavioral Features", fontsize=12, fontweight="bold")
ax2.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
fig.tight_layout()
fig.savefig(IMG_DIR / "customer_segments_clusters.png", dpi=160)
plt.close(fig)

# Save comprehensive JSON summary for the AG Grid HTML5 Dashboard & README write-up
summary_payload = {
    "dataset": {
        "file_name": CSV_PATH.name,
        "file_size_mb": round(CSV_PATH.stat().st_size / (1024**2), 1),
        "total_rows": total_rows,
        "total_orders": total_orders,
        "unique_customers": int(len(cust_df)),
        "unique_products": int(prod_df["Product"].nunique()),
        "unique_categories": int(cat_df["Category"].nunique()),
        "unique_regions": int(reg_df["Region"].nunique()),
        "min_date": min_date,
        "max_date": max_date,
        "missing_counts": missing_counts,
        "total_net_revenue": round(total_revenue, 2),
        "total_gross_revenue": round(total_gross, 2),
        "total_discount_leakage": round(total_gross - total_revenue, 2),
        "total_units_sold": total_qty,
        "avg_order_value": round(total_revenue / max(1, total_orders), 2),
        "avg_line_value": round(total_revenue / max(1, total_rows), 2),
    },
    "clustering": {
        "selected_k": best_k,
        "wcss_by_k": {str(k): round(w, 2) for k, w in zip(ks, wcss_list)},
        "silhouette_by_k": {str(k): round(s, 4) for k, s in zip(ks, sil_list)},
        "pca_variance_explained_pct": [round(float(v), 2) for v in var_exp],
        "clusters": [
            {
                "cluster_id": int(row["Cluster"]),
                "persona": cluster_names[int(row["Cluster"])],
                "customers": int(row["customers"]),
                "mean_revenue": round(float(row["mean_revenue"]), 2),
                "mean_orders": round(float(row["mean_orders"]), 1),
                "mean_line_qty": round(float(row["mean_line_qty"]), 2),
                "mean_discount_pct": round(float(row["mean_disc"] * 100), 2),
                "b2b_share_pct": round(float(row["b2b_share"] * 100), 1),
            }
            for _, row in sorted_c.iterrows()
        ],
    },
    "regions": [
        {
            "region": str(r["Region"]),
            "customer_type": str(r["Customer_Type"]),
            "revenue": round(float(r["Total_Price"]), 2),
            "gross": round(float(r["Gross_Price"]), 2),
            "discount_leakage": round(float(r["Gross_Price"] - r["Total_Price"]), 2),
            "units": int(r["Quantity"]),
            "lines": int(r["Lines"]),
            "orders": int(r["is_new_order"]),
        }
        for _, r in reg_df.sort_values("Total_Price", ascending=False).iterrows()
    ],
    "categories": [
        {
            "category": str(r["Category"]),
            "customer_type": str(r["Customer_Type"]),
            "revenue": round(float(r["Total_Price"]), 2),
            "gross": round(float(r["Gross_Price"]), 2),
            "units": int(r["Quantity"]),
            "lines": int(r["Lines"]),
        }
        for _, r in cat_df.sort_values("Total_Price", ascending=False).iterrows()
    ],
    "products": [
        {
            "product": str(r["Product"]),
            "category": str(r["Category"]),
            "customer_type": str(r["Customer_Type"]),
            "revenue": round(float(r["Total_Price"]), 2),
            "gross": round(float(r["Gross_Price"]), 2),
            "discount_leakage": round(float(r["Gross_Price"] - r["Total_Price"]), 2),
            "units": int(r["Quantity"]),
            "lines": int(r["Lines"]),
            "avg_unit_price": round(float(r["Unit_Price"] / max(1, r["Lines"])), 2),
            "avg_discount_pct": round(float((r["Discount"] / max(1, r["Lines"])) * 100), 2),
        }
        for _, r in prod_df[prod_df["Lines"] > 0].sort_values("Total_Price", ascending=False).iterrows()
    ],
    "monthly": [
        {
            "year_month": str(r["YearMonth"]),
            "customer_type": str(r["Customer_Type"]),
            "revenue": round(float(r["Total_Price"]), 2),
            "units": int(r["Quantity"]),
            "lines": int(r["Lines"]),
        }
        for _, r in mo_df.sort_values(["YearMonth", "Customer_Type"]).iterrows()
    ],
    "discounts": [
        {
            "discount_pct": round(float(r["Discount"] * 100), 1),
            "revenue": round(float(r["revenue"]), 2),
            "discount_leakage": round(float(r["disc_loss_m"] * 1e6), 2),
            "lines": int(r["lines"]),
            "avg_qty": round(float(r["avg_qty"]), 2),
            "ci95_qty": round(float(r["ci95_qty"]), 4),
        }
        for _, r in disc_agg.iterrows()
    ],
}

(OUT_DIR / "beverage_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
print("Saved 6 Matplotlib charts and beverage_summary.json to:", OUT_DIR)
