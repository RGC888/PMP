import os
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

az.rcParams["stats.ci_prob"] = 0.95  # am setat HDI la 95%
pd.set_option('display.float_format', lambda x: f'{x:.2f}') 

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

x_bar = data.mean()
s_std = data.std(ddof=1) # ddof=1 petru deviația standard

def main():
    print("--- Frequentist Estimates ---")
    print(f"Sample mean = {x_bar:.2f} dB")
    print(f"Sample std  = {s_std:.2f} dB")
    print("------------------------------\n")

    # a), b)
    print("--- Building Weak Prior Model (a, b) ---")
    with pm.Model() as weak_model:
        #priors
        # pentru mu ~ N(x_bar, 10^2), setam mu=x_bar si sigma=10
        mu = pm.Normal("mu", mu=x_bar, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        # pt b)
        trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, cores=1)
        summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"])

    print("\nPosterior Summaries (Weak Prior):")
    print(summary_weak)
    print("------------------------------\n")

    # d)
    print("--- Building Strong Prior Model (d) ---")
    with pm.Model() as strong_model:
        # d)definim priori puternici: mu ~ N(50, 1^2)
        mu = pm.Normal("mu", mu=50, sigma=1)  
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        trace_strong = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, cores=1)
        summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"])

    print("\nPosterior Summaries (Strong Prior):")
    print(summary_strong)
    print("------------------------------\n")

    print("Generating plots...")
    outdir = os.path.join(os.path.dirname(__file__), "lab7")
    os.makedirs(outdir, exist_ok=True)

    az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Weak Prior (b)", fontsize=14, y=1.02)
    out1 = os.path.join(outdir, "lab7_posterior_weak.png")
    plt.savefig(out1, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Strong Prior (d)", fontsize=14, y=1.02)
    out2 = os.path.join(outdir, "lab7_posterior_strong.png")
    plt.savefig(out2, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out2}")

    print("Done.")


if __name__ == "__main__":
    main()