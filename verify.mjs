import assert from "node:assert/strict";
import fs from "node:fs";

const html = fs.readFileSync(new URL("./index.html", import.meta.url), "utf8");
const demo = fs.readFileSync(new URL("./demos/executive-reset-pack.html", import.meta.url), "utf8");
const bevDemo = fs.readFileSync(new URL("./demos/beverage-analytics-dashboard.html", import.meta.url), "utf8");
const readme = fs.readFileSync(new URL("./README.md", import.meta.url), "utf8");

// 1. Verify color-scheme meta and CSS light-dark + fallback support on all pages
for (const doc of [html, demo, bevDemo]) {
  assert.match(doc, /<meta name="color-scheme" content="light dark">/);
  assert.match(doc, /color-scheme:\s*light dark/);
  assert.match(doc, /@media\s*\(prefers-color-scheme:\s*dark\)/);
  assert.equal((doc.match(/<h1\b/g) || []).length, 1, "Must have exactly one <h1>");
  assert.match(doc, /<search\b/);
}

// 2. Verify 7 case study cards on main portfolio, AG Grid in beverage dashboard, and 7 Matplotlib PNG charts
assert.equal((html.match(/<article class="case-card"/g) || []).length, 7, "Must have 7 case studies");
assert.match(html, /data-filter="stats"/);
assert.match(bevDemo, /ag-grid-community/);
assert.match(bevDemo, /agGrid\.createGrid/);

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
}

// 3. Verify no headings inside <summary> (accessibility rule)
for (const [, summaryInner] of html.matchAll(/<summary>([\s\S]*?)<\/summary>/g)) {
  assert.doesNotMatch(summaryInner, /<h[1-6]\b/i, "Do not place headings inside <summary>");
}

// 4. Ensure no internal employer/local username paths leaked into public portfolio
assert.doesNotMatch(html + demo + bevDemo + readme, /saval01|Blackhawk Network|\bBHN\b/i, "Sanitize internal paths/company names");

// 5. Verify README has private repo setup instructions
assert.match(readme, /gh repo create .* --private/);

console.log("OK: All portfolio and demo verification checks passed.");
