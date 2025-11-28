import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

scenarios = {
    'Y=0, θ=0.2': {'Y': 0, 'theta': 0.2},
    'Y=5, θ=0.2': {'Y': 5, 'theta': 0.2},
    'Y=10, θ=0.2': {'Y': 10, 'theta': 0.2},
    'Y=0, θ=0.5': {'Y': 0, 'theta': 0.5},
    'Y=5, θ=0.5': {'Y': 5, 'theta': 0.5},
    'Y=10, θ=0.5': {'Y': 10, 'theta': 0.5},
}

traces = {}
posterior_predictive_samples = {}

for name, params in scenarios.items():
    Y_obs = params['Y']
    theta_val = params['theta']
    
    with pm.Model() as model:
        
        n_unbounded = pm.Poisson('n_unbounded', mu=10)
        n = pm.Deterministic('n', n_unbounded)
        pm.Potential('n_bounded', pm.math.switch(n < Y_obs, -np.inf, 0))

        y_obs = pm.Binomial('y_obs', n=n, p=theta_val, observed=Y_obs)

        y_pred = pm.Binomial('y_pred', n=n, p=theta_val)

        step = pm.Metropolis(vars=[n_unbounded])
        trace = pm.sample(2000, tune=2000, step=step, chains=2, cores=1, random_seed=2025, progressbar=False)
        
        ppc = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=2025, progressbar=False)

        traces[name] = trace
        posterior_predictive_samples[name] = ppc.posterior_predictive['y_pred'].values.flatten()

#afisam rez pe o singura fereasta
fig_posterior, axes_posterior = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
axes_posterior = axes_posterior.flatten()

for i, (name, trace) in enumerate(traces.items()):
    az.plot_posterior(trace, var_names=['n'], ax=axes_posterior[i])
    axes_posterior[i].set_title(f'Posterior of n for {name}')

fig_posterior.suptitle('Posterior distribution of n for different scenarios', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('lab9/posterior_n.png')
print("Posterior plots for n saved to lab9/posterior_n.png")

#C

fig_pred, axes_pred = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
axes_pred = axes_pred.flatten()

for i, (name, samples) in enumerate(posterior_predictive_samples.items()):
    az.plot_dist(samples, ax=axes_pred[i])
    axes_pred[i].set_title(f'Posterior Predictive of Y* for {name}')

fig_pred.suptitle('Posterior predictive distribution of Y* for different scenarios', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('lab9/posterior_predictive_y_star.png')
print("Posterior predictive plots for Y* saved to lab9/posterior_predictive_y_star.png")

plt.show()

