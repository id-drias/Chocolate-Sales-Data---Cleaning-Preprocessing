# Chocolate Sales Analysis - Complete Script
# Run with: python run_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("="*70)
print("CHOCOLATE SALES ANALYSIS - COMPLETE PIPELINE")
print("="*70)

# ============================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================
print("\n[1/5] Loading and cleaning data...")

df = pd.read_csv('Chocolate Sales (2).csv')
print(f"  Loaded {len(df)} records")

# Clean Amount column
df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)

# Parse Date
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract date components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter

# Strip whitespace
for col in ['Sales Person', 'Country', 'Product']:
    df[col] = df[col].str.strip()

# Save cleaned data
df.to_csv('Chocolate_Sales_Cleaned.csv', index=False)
print("  Cleaned data saved to Chocolate_Sales_Cleaned.csv")

# ============================================================
# STEP 2: DATA SUMMARY
# ============================================================
print("\n[2/5] Data Summary...")
print(f"  Shape: {df.shape}")
print(f"  Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"  Total Revenue: ${df['Amount'].sum():,.0f}")
print(f"  Countries: {df['Country'].nunique()}")
print(f"  Products: {df['Product'].nunique()}")
print(f"  Sales People: {df['Sales Person'].nunique()}")

# ============================================================
# STEP 3: SALES PREDICTION (REGRESSION)
# ============================================================
print("\n[3/5] Training Sales Prediction Models...")

# Encode categorical variables
le_country = LabelEncoder()
le_product = LabelEncoder()
le_salesperson = LabelEncoder()
le_weekday = LabelEncoder()

df_ml = df.copy()
df_ml['Country_Encoded'] = le_country.fit_transform(df_ml['Country'])
df_ml['Product_Encoded'] = le_product.fit_transform(df_ml['Product'])
df_ml['SalesPerson_Encoded'] = le_salesperson.fit_transform(df_ml['Sales Person'])
df_ml['Weekday_Encoded'] = le_weekday.fit_transform(df_ml['Weekday'])

features = ['Country_Encoded', 'Product_Encoded', 'SalesPerson_Encoded', 
            'Month', 'Quarter', 'Weekday_Encoded', 'Year', 'Boxes Shipped']

X = df_ml[features]
y = df_ml['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    print(f"  {name:20} R2: {r2:.4f} | RMSE: ${rmse:,.0f}")

# ============================================================
# STEP 4: PRODUCT CLASSIFICATION
# ============================================================
print("\n[4/5] Training Product Classification Models...")

product_stats = df.groupby('Product').agg({
    'Amount': ['sum', 'mean', 'count'],
    'Boxes Shipped': ['sum', 'mean']
}).reset_index()
product_stats.columns = ['Product', 'Total_Revenue', 'Avg_Revenue', 'Transactions', 
                         'Total_Boxes', 'Avg_Boxes']
product_stats['Revenue_Per_Box'] = product_stats['Total_Revenue'] / product_stats['Total_Boxes']

q33 = product_stats['Total_Revenue'].quantile(0.33)
q66 = product_stats['Total_Revenue'].quantile(0.66)

def assign_tier(revenue):
    if revenue >= q66: return 'High'
    elif revenue >= q33: return 'Medium'
    else: return 'Low'

product_stats['Performance_Tier'] = product_stats['Total_Revenue'].apply(assign_tier)

X_class = product_stats[['Avg_Revenue', 'Transactions', 'Avg_Boxes', 'Revenue_Per_Box']]
y_class = product_stats['Performance_Tier']

le_tier = LabelEncoder()
y_class_encoded = le_tier.fit_transform(y_class)

scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class_scaled, y_class_encoded, test_size=0.3, random_state=42
)

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"  {name:20} Accuracy: {acc:.2%}")

# ============================================================
# STEP 5: GENERATE VISUALIZATIONS
# ============================================================
print("\n[5/5] Generating visualizations...")

# Regression visualization
rf_model = models['Random Forest']
y_pred_best = rf_model.predict(X_test)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature importance
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)
axes[0, 0].barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue')
axes[0, 0].set_title('Feature Importance - Sales Prediction')

# Actual vs Predicted
axes[0, 1].scatter(y_test, y_pred_best, alpha=0.5, c='steelblue', s=20)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual ($)')
axes[0, 1].set_ylabel('Predicted ($)')
axes[0, 1].set_title(f'Actual vs Predicted (R2={r2_score(y_test, y_pred_best):.3f})')

# Residuals
residuals = y_test - y_pred_best
axes[1, 0].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='red', linestyle='--')
axes[1, 0].set_title('Residuals Distribution')

# Model comparison
results_df = pd.DataFrame(results)
colors = ['#45b7d1' if r > 0.5 else '#ff6b6b' for r in results_df['R2']]
axes[1, 1].barh(results_df['Model'], results_df['R2'], color=colors)
axes[1, 1].set_xlabel('R2 Score')
axes[1, 1].set_title('Model Comparison')

plt.tight_layout()
plt.savefig('ml_regression_analysis.png', dpi=150)
print("  Saved: ml_regression_analysis.png")

# Classification visualization
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

best_clf = classifiers['Random Forest']
y_pred_final = best_clf.predict(X_test_c)
cm = confusion_matrix(y_test_c, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_tier.classes_, yticklabels=le_tier.classes_, ax=axes2[0])
axes2[0].set_title('Confusion Matrix')

clf_feat_imp = pd.DataFrame({
    'Feature': ['Avg_Revenue', 'Transactions', 'Avg_Boxes', 'Revenue_Per_Box'],
    'Importance': best_clf.feature_importances_
}).sort_values('Importance', ascending=True)
axes2[1].barh(clf_feat_imp['Feature'], clf_feat_imp['Importance'], color='coral')
axes2[1].set_title('Classification Feature Importance')

plt.tight_layout()
plt.savefig('ml_classification_analysis.png', dpi=150)
print("  Saved: ml_classification_analysis.png")

plt.close('all')

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nBest Regression Model: Random Forest (R2 = {:.4f})".format(results_df.loc[results_df['R2'].idxmax(), 'R2']))
print("Top Predictor: Boxes Shipped")
print("\nFiles Generated:")
print("  - Chocolate_Sales_Cleaned.csv")
print("  - ml_regression_analysis.png")
print("  - ml_classification_analysis.png")
print("="*70)
