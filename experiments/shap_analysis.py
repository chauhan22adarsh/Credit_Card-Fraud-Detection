import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap

with open("best_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
X_test = saved["X_test"]
y_test = saved["y_test"]
feature_names = saved["feature_names"]
model_name = saved["model_name"]

print(f"Explaining {model_name} with SHAP...")

explainer = shap.TreeExplainer(model)

rng = np.random.default_rng(42)
fraud_idx = np.where(y_test == 1)[0]
normal_idx = np.where(y_test == 0)[0]
sample_idx = np.concatenate([
    fraud_idx,  # keep ALL fraud cases in the test set (only ~98 of them)
    rng.choice(normal_idx, size=2000 - len(fraud_idx), replace=False),
])
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

print(f"Computing SHAP values on a {len(X_sample)}-row sample "
      f"({y_sample.sum()} fraud, {len(y_sample)-y_sample.sum()} normal)...")

shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                   plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance — {model_name}")
plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=120)
plt.close()
print("Saved shap_importance_bar.png")

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.title(f"SHAP Summary — {model_name}")
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=120)
plt.close()
print("Saved shap_summary_beeswarm.png")

fraud_positions_in_sample = np.where(y_sample == 1)[0]
if len(fraud_positions_in_sample) > 0:
    example_idx = fraud_positions_in_sample[0]
    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[example_idx],
        X_sample[example_idx], feature_names=feature_names,
        matplotlib=True, show=False,
    )
    plt.title(f"SHAP Explanation — a single real fraud case", fontsize=10)
    plt.tight_layout()
    plt.savefig("shap_single_case.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved shap_single_case.png")

# Print the top 10 features by mean |SHAP value| as a plain-text summary
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
print("\nTop 10 features by mean |SHAP value|:")
for i in top10_idx:
    print(f"  {feature_names[i]:16s} {mean_abs_shap[i]:.4f}")
