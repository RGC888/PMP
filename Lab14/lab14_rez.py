import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

df = pd.read_csv("Lab14/date_colesterol.csv")
df.columns = ["exercise", "cholesterol"]

x = df["exercise"].values
y = df["cholesterol"].values
N = len(y)

def fit_mixture(K):
    with pm.Model() as model:

        w = pm.Dirichlet("w", a=np.ones(K))

        alpha = pm.Normal(
            "alpha",
            mu=np.linspace(y.min(), y.max(), K),
            sigma=50,
            shape=K,
            transform=pm.distributions.transforms.ordered
        )

        beta = pm.Normal("beta", mu=0, sigma=10, shape=K)
        gamma = pm.Normal("gamma", mu=0, sigma=5, shape=K)

        sigma = pm.HalfNormal("sigma", sigma=10, shape=K)

        mu = alpha[:, None] + beta[:, None] * x + gamma[:, None] * x**2

        logp_components = (
            -0.5 * pt.log(2 * np.pi * sigma[:, None]**2)
            - 0.5 * ((y - mu) / sigma[:, None])**2
        )
        
        logp_weighted = pt.log(w)[:, None] + logp_components
        
        logp_mixture = pm.math.logsumexp(logp_weighted, axis=0)
        
        pm.Deterministic("log_likelihood", logp_mixture)
        
        pm.Potential("y_obs", logp_mixture.sum())

        idata = pm.sample(
            1000,
            tune=1000,
            target_accept=0.95,
            random_seed=123,
            return_inferencedata=True
        )

    return model, idata

if __name__ == '__main__':
    plt.scatter(x, y, alpha=0.4)
    plt.xlabel("Ore exercitii / saptamana")
    plt.ylabel("Colesterol")
    plt.title("Exercitii fizice vs Colesterol")
    plt.show()
    
    clusters = [3, 4, 5]
    models = []
    idatas = []

    for K in clusters:
        print(f"\nFitting model with K = {K}")
        model, idata = fit_mixture(K)
        models.append(model)
        idatas.append(idata)

    for K, idata in zip(clusters, idatas):
        print(f"\n===== SUMMARY K = {K} =====")
        print(
            az.summary(
                idata,
                var_names=["w", "alpha", "beta", "gamma", "sigma"],
                round_to=2
            )
        )

    print("\n===== WAIC COMPARISON =====")
    
    for idata in idatas:
        if "log_likelihood" in idata.posterior:
            idata.add_groups({"log_likelihood": {"y_obs": idata.posterior["log_likelihood"]}})
    
    comp = az.compare(
        dict(zip([str(k) for k in clusters], idatas)),
        ic="waic",
        method="BB-pseudo-BMA",
        scale="deviance"
    )

    print("\n===== WAIC COMPARISON =====")
    print(comp)

    az.plot_compare(comp)
    plt.show()

    x_grid = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for idx, (K, idata) in enumerate(zip(clusters, idatas)):
        posterior = idata.posterior.stack(samples=("chain", "draw"))

        ax[idx].scatter(x, y, alpha=0.25)

        for _ in range(50):
            i = np.random.randint(posterior.samples.size)
            a = posterior["alpha"][:, i].values
            b = posterior["beta"][:, i].values
            g = posterior["gamma"][:, i].values
            w_sample = posterior["w"][:, i].values

            y_hat = np.sum(
                w_sample[:, None] *
                (a[:, None] + b[:, None]*x_grid + g[:, None]*x_grid**2),
                axis=0
            )
            ax[idx].plot(x_grid, y_hat, "C0", alpha=0.1)

        ax[idx].set_title(f"K = {K}")
        ax[idx].set_xlabel("Ore exercitii")

    ax[0].set_ylabel("Colesterol")
    plt.show()
