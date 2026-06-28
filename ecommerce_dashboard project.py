"""
========================================================
 E-COMMERCE SALES ANALYTICS DASHBOARD
 Author  : Sanyam Jain
 Tools   : Python | Pandas | NumPy | Matplotlib | Seaborn | OpenPyXL
 Dataset : Synthetically generated realistic e-commerce data
========================================================

PROJECT OVERVIEW
----------------
This project simulates a real-world business analytics dashboard
for an e-commerce company. It covers the full data pipeline:
  1. Data Generation  → realistic synthetic sales data
  2. Data Cleaning    → handling nulls, type fixing, deduplication
  3. EDA              → exploratory analysis with statistics
  4. KPI Computation  → revenue, growth, retention, churn metrics
  5. Visualisation    → 6-panel executive dashboard
  6. Export           → Excel report with multiple sheets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1 — GENERATE REALISTIC SYNTHETIC DATA
# ─────────────────────────────────────────────
"""
WHY SYNTHETIC DATA?
In real jobs you won't always have a ready dataset.
Knowing how to generate realistic data shows you understand
distributions, seasonality, and business logic — a rare skill.
"""

np.random.seed(42)          # Ensures reproducibility
N_ORDERS = 5000             # Simulate 5,000 orders

# Product catalogue — realistic categories and prices
CATEGORIES = {
    "Electronics":   (3000, 25000),
    "Fashion":       (500,  5000),
    "Home & Kitchen":(800,  8000),
    "Books":         (150,  1200),
    "Sports":        (400,  6000),
    "Beauty":        (200,  3000),
}

CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad",
          "Chennai", "Pune", "Kolkata", "Ahmedabad"]

PAYMENT  = ["UPI", "Credit Card", "Debit Card", "COD", "Wallet"]

# ── Date range: Jan 2023 – Dec 2023 (full year) ──
start_date = datetime(2023, 1, 1)
end_date   = datetime(2023, 12, 31)
date_range = (end_date - start_date).days

# ── Simulate seasonal order volume (more orders in Oct-Dec festive season) ──
# We use a weighted random day selection
day_indices = np.arange(date_range + 1)
# Weight: higher for festive months (Oct=274, Nov=305, Dec=335)
weights = np.ones(date_range + 1)
weights[274:] *= 2.5          # Festive season boost
weights = weights / weights.sum()

order_days = np.random.choice(day_indices, size=N_ORDERS, p=weights)
order_dates = [start_date + timedelta(days=int(d)) for d in order_days]

# ── Build the DataFrame ──
categories = np.random.choice(list(CATEGORIES.keys()), size=N_ORDERS)

prices = []
for cat in categories:
    low, high = CATEGORIES[cat]
    prices.append(round(np.random.uniform(low, high), 2))

# Discount: 0–30% off, Electronics more discounted during festive
discounts = np.random.uniform(0, 0.30, size=N_ORDERS)

quantities = np.random.choice([1, 1, 1, 2, 2, 3], size=N_ORDERS)   # mostly qty=1

revenue = [round(p * q * (1 - d), 2)
           for p, q, d in zip(prices, quantities, discounts)]

# Customer IDs: 1000 unique customers → repeat purchases possible
customer_ids = np.random.randint(1001, 2001, size=N_ORDERS)

df = pd.DataFrame({
    "order_id":    range(10001, 10001 + N_ORDERS),
    "order_date":  order_dates,
    "customer_id": customer_ids,
    "category":    categories,
    "city":        np.random.choice(CITIES, size=N_ORDERS),
    "payment":     np.random.choice(PAYMENT, size=N_ORDERS),
    "quantity":    quantities,
    "unit_price":  prices,
    "discount_pct":np.round(discounts * 100, 1),
    "revenue":     revenue,
    "rating":      np.round(np.random.uniform(2.5, 5.0, size=N_ORDERS), 1),
})

# Introduce ~2% nulls in rating (realistic — not all customers rate)
null_idx = np.random.choice(df.index, size=int(N_ORDERS * 0.02), replace=False)
df.loc[null_idx, "rating"] = np.nan

# Introduce ~1% duplicate rows (realistic data quality issue)
dup_idx = np.random.choice(df.index, size=int(N_ORDERS * 0.01), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

print(f"Raw data shape: {df.shape}")

# ─────────────────────────────────────────────
# STEP 2 — DATA CLEANING
# ─────────────────────────────────────────────
"""
WHAT WE DO & WHY:
- Drop duplicates      → avoids inflated revenue numbers
- Fill missing ratings → median fill is safer than mean (less affected by outliers)
- Ensure correct types → date must be datetime for time-series operations
- Add derived columns  → month, week, day_of_week for time analysis
"""

df.drop_duplicates(subset=["order_id"], inplace=True)
df["rating"].fillna(df["rating"].median(), inplace=True)
df["order_date"] = pd.to_datetime(df["order_date"])
df["month"]      = df["order_date"].dt.month
df["month_name"] = df["order_date"].dt.strftime("%b")
df["quarter"]    = df["order_date"].dt.quarter
df["day_of_week"]= df["order_date"].dt.day_name()
df["week"]       = df["order_date"].dt.isocalendar().week.astype(int)

print(f"Clean data shape: {df.shape}")
print(f"Nulls remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# STEP 3 — KPI COMPUTATION
# ─────────────────────────────────────────────
"""
KPIs (Key Performance Indicators) are the metrics business
stakeholders care about most. Always compute these first.
"""

total_revenue   = df["revenue"].sum()
total_orders    = df["order_id"].nunique()
unique_customers= df["customer_id"].nunique()
avg_order_value = total_revenue / total_orders
avg_rating      = df["rating"].mean()

# Month-over-month revenue
monthly_rev = df.groupby("month")["revenue"].sum()
mom_growth  = monthly_rev.pct_change().mean() * 100   # avg MoM growth %

# Repeat purchase rate (customers with > 1 order)
cust_orders      = df.groupby("customer_id")["order_id"].nunique()
repeat_customers = (cust_orders > 1).sum()
repeat_rate      = repeat_customers / unique_customers * 100

print(f"\n{'='*45}")
print(f"  BUSINESS KPIs — FY 2023")
print(f"{'='*45}")
print(f"  Total Revenue     : ₹{total_revenue:,.0f}")
print(f"  Total Orders      : {total_orders:,}")
print(f"  Unique Customers  : {unique_customers:,}")
print(f"  Avg Order Value   : ₹{avg_order_value:,.0f}")
print(f"  Avg Rating        : {avg_rating:.2f} / 5.0")
print(f"  Repeat Rate       : {repeat_rate:.1f}%")
print(f"  Avg MoM Growth    : {mom_growth:.1f}%")
print(f"{'='*45}\n")

# ─────────────────────────────────────────────
# STEP 4 — ANALYSIS DATAFRAMES
# ─────────────────────────────────────────────

# A. Monthly revenue trend
monthly = (df.groupby(["month", "month_name"])["revenue"]
             .sum().reset_index()
             .sort_values("month"))

# B. Revenue by category
cat_rev = (df.groupby("category")["revenue"]
             .sum().sort_values(ascending=False).reset_index())

# C. City-wise revenue
city_rev = (df.groupby("city")["revenue"]
              .sum().sort_values(ascending=False).reset_index())

# D. Payment method share
pay_share = df["payment"].value_counts(normalize=True) * 100

# E. Day-of-week orders
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow = (df.groupby("day_of_week")["order_id"]
         .count().reindex(dow_order).reset_index())
dow.columns = ["day", "orders"]

# F. Rating distribution by category
rat_cat = df.groupby("category")["rating"].mean().sort_values(ascending=False)

# ─────────────────────────────────────────────
# STEP 5 — DASHBOARD VISUALISATION
# ─────────────────────────────────────────────
"""
We build a 6-panel dashboard that tells a complete business story:
  Panel 1: Monthly revenue trend (line chart)
  Panel 2: Revenue by category  (horizontal bar)
  Panel 3: City-wise revenue    (bar chart)
  Panel 4: Payment method share (donut chart)
  Panel 5: Orders by day of week(bar chart)
  Panel 6: Avg rating by category (bar chart)
"""

# ── Colour palette ──
PALETTE = ["#1a237e","#283593","#3949ab","#5c6bc0","#7986cb","#9fa8da"]
BG      = "#f4f6fb"
CARD_BG = "#ffffff"

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(BG)

# Title banner
fig.text(0.5, 0.97,
         "E-Commerce Sales Analytics Dashboard — FY 2023",
         ha="center", va="top", fontsize=20, fontweight="bold",
         color="#1a237e", fontfamily="DejaVu Sans")

fig.text(0.5, 0.945,
         f"Total Revenue: ₹{total_revenue/1e7:.2f} Cr  |  "
         f"Orders: {total_orders:,}  |  "
         f"Avg Order Value: ₹{avg_order_value:,.0f}  |  "
         f"Repeat Rate: {repeat_rate:.1f}%  |  "
         f"Avg Rating: {avg_rating:.2f}★",
         ha="center", va="top", fontsize=11, color="#555",
         fontfamily="DejaVu Sans")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.42, wspace=0.35,
                       top=0.91, bottom=0.06,
                       left=0.06, right=0.97)

# ── Helper ──
def style_ax(ax, title):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color="#1a237e", pad=10)
    ax.tick_params(labelsize=9, colors="#444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd")
    ax.grid(axis="y", color="#eee", linewidth=0.8)

# ── Panel 1: Monthly Revenue ──
ax1 = fig.add_subplot(gs[0, :2])   # spans 2 columns
ax1.set_facecolor(CARD_BG)
ax1.fill_between(monthly["month"], monthly["revenue"],
                 alpha=0.15, color="#3949ab")
ax1.plot(monthly["month"], monthly["revenue"],
         color="#1a237e", linewidth=2.5, marker="o",
         markersize=7, markerfacecolor="#ff6f00")
ax1.set_xticks(monthly["month"])
ax1.set_xticklabels(monthly["month_name"], fontsize=9)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
style_ax(ax1, "📈  Monthly Revenue Trend")
ax1.set_xlabel("Month", fontsize=9)
ax1.set_ylabel("Revenue (₹ Millions)", fontsize=9)

# Annotate peak month
peak_m = monthly.loc[monthly["revenue"].idxmax()]
ax1.annotate(f"Peak\n₹{peak_m['revenue']/1e6:.1f}M",
             xy=(peak_m["month"], peak_m["revenue"]),
             xytext=(peak_m["month"]-1.2, peak_m["revenue"]*0.95),
             fontsize=8, color="#e65100",
             arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.2))

# ── Panel 2: Revenue by Category ──
ax2 = fig.add_subplot(gs[0, 2])
bars = ax2.barh(cat_rev["category"], cat_rev["revenue"],
                color=PALETTE, edgecolor="white", height=0.6)
ax2.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.0f}M"))
for bar, val in zip(bars, cat_rev["revenue"]):
    ax2.text(bar.get_width() + 1e5, bar.get_y() + bar.get_height()/2,
             f"₹{val/1e6:.1f}M", va="center", fontsize=8, color="#333")
style_ax(ax2, "🏷  Revenue by Category")
ax2.grid(axis="x", color="#eee"); ax2.grid(axis="y", visible=False)
ax2.set_xlabel("Revenue (₹ Millions)", fontsize=9)

# ── Panel 3: City Revenue ──
ax3 = fig.add_subplot(gs[1, 0])
colors3 = [PALETTE[0] if i == 0 else PALETTE[3]
           for i in range(len(city_rev))]
ax3.bar(city_rev["city"], city_rev["revenue"],
        color=colors3, edgecolor="white")
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.0f}M"))
ax3.set_xticklabels(city_rev["city"], rotation=30, ha="right", fontsize=8)
style_ax(ax3, "🏙  Revenue by City")
ax3.set_ylabel("Revenue (₹ Millions)", fontsize=9)

# ── Panel 4: Payment Methods ──
ax4 = fig.add_subplot(gs[1, 1])
wedges, texts, autotexts = ax4.pie(
    pay_share.values, labels=pay_share.index,
    autopct="%1.1f%%", colors=PALETTE,
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
for t in texts:    t.set_fontsize(9)
for t in autotexts:t.set_fontsize(8); t.set_color("white")
ax4.set_title("💳  Payment Method Share", fontsize=12,
              fontweight="bold", color="#1a237e", pad=10)

# ── Panel 5: Orders by Day of Week ──
ax5 = fig.add_subplot(gs[1, 2])
weekend_colors = [PALETTE[0] if d not in ["Saturday","Sunday"]
                  else "#ff6f00" for d in dow["day"]]
ax5.bar(dow["day"], dow["orders"], color=weekend_colors, edgecolor="white")
ax5.set_xticklabels([d[:3] for d in dow["day"]], rotation=0, fontsize=9)
style_ax(ax5, "📅  Orders by Day of Week")
ax5.set_ylabel("Number of Orders", fontsize=9)
legend_els = [mpatches.Patch(color=PALETTE[0], label="Weekday"),
              mpatches.Patch(color="#ff6f00",   label="Weekend")]
ax5.legend(handles=legend_els, fontsize=8, loc="upper right")

# Save dashboard
out_path = "/mnt/user-data/outputs/ecommerce_dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✅ Dashboard saved → {out_path}")

# ─────────────────────────────────────────────
# STEP 6 — EXPORT EXCEL REPORT
# ─────────────────────────────────────────────
"""
Export an Excel workbook with multiple sheets — this is exactly
what a Data Analyst delivers to a business team.
"""

excel_path = "/mnt/user-data/outputs/ecommerce_report.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    monthly.to_excel(writer, sheet_name="Monthly Revenue", index=False)
    cat_rev.to_excel(writer, sheet_name="Category Revenue", index=False)
    city_rev.to_excel(writer, sheet_name="City Revenue", index=False)
    pay_share.reset_index().to_excel(writer, sheet_name="Payment Methods", index=False)
    dow.to_excel(writer, sheet_name="Day of Week", index=False)
    rat_cat.reset_index().to_excel(writer, sheet_name="Ratings", index=False)

    # KPI summary sheet
    kpi_df = pd.DataFrame({
        "KPI":   ["Total Revenue","Total Orders","Unique Customers",
                  "Avg Order Value","Avg Rating","Repeat Rate %","Avg MoM Growth %"],
        "Value": [f"₹{total_revenue:,.0f}", f"{total_orders:,}",
                  f"{unique_customers:,}", f"₹{avg_order_value:,.0f}",
                  f"{avg_rating:.2f}", f"{repeat_rate:.1f}%",
                  f"{mom_growth:.1f}%"]
    })
    kpi_df.to_excel(writer, sheet_name="KPI Summary", index=False)

print(f"✅ Excel report saved → {excel_path}")
print("\n🎉 Project complete!")
