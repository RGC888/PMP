import numpy as np
import scipy.stats as stats
import arviz as az
import matplotlib.pyplot as plt

k = 180      
T = 10       

alpha_prior = 2
beta_prior = 1

# a)
alpha_post = alpha_prior + k
beta_post = beta_prior + T

print("=== (a) Posterior distribution ===")
print(f"Posterior: Gamma(α' = {alpha_post}, β' = {beta_post})")
print(f"Mean (α'/β') = {alpha_post / beta_post:.3f} calls/hour")

# b)
posterior_samples = stats.gamma(a=alpha_post, scale=1/beta_post).rvs(size=100000)
hdi_94 = az.hdi(posterior_samples, hdi_prob=0.94)

print("\n=== (b) 94% HDI ===")
print(f"94% HDI for λ: [{hdi_94[0]:.3f}, {hdi_94[1]:.3f}] calls/hour")

# c)
mode_lambda = (alpha_post - 1) / beta_post if alpha_post > 1 else 0
print("\n=== (c) Most probable value ===")
print(f"Mode of posterior (MAP estimate): λ = {mode_lambda:.3f} calls/hour")

# --- Visualization ---
x = np.linspace(10, 26, 500)
posterior_pdf = stats.gamma.pdf(x, a=alpha_post, scale=1/beta_post)

plt.figure(figsize=(9, 5))
plt.plot(x, posterior_pdf, label='Posterior Gamma PDF')
plt.axvline(mode_lambda, color='red', linestyle='--', label=f'Mode = {mode_lambda:.2f}')
plt.axvline(hdi_94[0], color='gray', linestyle=':', label='94% HDI bounds')
plt.axvline(hdi_94[1], color='gray', linestyle=':')
plt.title('Posterior distribution of λ (call rate per hour)')
plt.xlabel('λ (calls/hour)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
