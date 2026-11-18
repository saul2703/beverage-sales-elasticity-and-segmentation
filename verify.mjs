import assert from "node:assert/strict";
import fs from "node:fs";

const html = fs.readFileSync(new URL("./index.html", import.meta.url), "utf8");
const readme = fs.readFileSync(new URL("./README.md", import.meta.url), "utf8");
const summary = JSON.parse(fs.readFileSync(new URL("./data/beverage_summary.json", import.meta.url), "utf8"));

// 1. Verify color-scheme meta, CSS light-dark + fallback support, semantic landmarks, and AG Grid
assert.match(html, /<meta name="color-scheme" content="light dark">/);
assert.match(html, /color-scheme:\s*light dark/);
assert.match(html, /@media\s*\(prefers-color-scheme:\s*dark\)/);
assert.equal((html.match(/<h1\b/g) || []).length, 1, "Must have exactly one <h1>");
assert.match(html, /<search\b/);
assert.match(html, /ag-grid-community/);
assert.match(html, /agGrid\.createGrid/);

// 2. Verify dataset integrity (8,999,910 rows, 3,000,000 orders, 10,000 customers)
assert.equal(summary.dataset.total_rows, 8999910);
assert.equal(summary.dataset.total_orders, 3000000);
assert.equal(summary.dataset.unique_customers, 10000);

// 3. Verify all 7 Matplotlib PNG charts exist
for (const img of [
  "dashboard_overview_preview.png",
  "regional_revenue_b2b_b2c.png",
  "category_product_performance.png",
  "monthly_sales_trend.png",
  "discount_price_elasticity.png",
  "elbow_silhouette_evaluation.png",
  "customer_segments_clusters.png",
]) {
  assert.ok(fs.existsSync(new URL(`./images/${img}`, import.meta.url)), `Missing chart: images/${img}`);
  assert.match(readme, new RegExp(`images/${img}`));
}

// 4. Verify Kaggle dataset citation link in both README.md and index.html
const kaggleUrl = "https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales/data";
assert.ok(readme.includes(kaggleUrl), "README.md must link to the Kaggle dataset");
assert.ok(html.includes(kaggleUrl), "index.html must link to the Kaggle dataset");

console.log("OK: Beverage analytics dashboard, Matplotlib charts, and README verification passed.");
