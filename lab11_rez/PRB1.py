import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'Lab11', 'Prices.csv')
    df = pd.read_csv(csv_path)

    y = df['Price'].values 
    x1 = df['Speed'].values  
    x2 = np.log(df['HardDrive'].values) 

    print("Data loaded successfully!")
    print(f"Number of samples: {len(y)}")
    print(f"\nSummary statistics:")
    print(f"Price: mean={y.mean():.2f}, std={y.std():.2f}")
    print(f"Speed: mean={x1.mean():.2f}, std={x1.std():.2f}")
    print(f"Log(HardDrive): mean={x2.mean():.2f}, std={x2.std():.2f}")

   # a)
    print("\n" + "="*60)
    print("Part a) Building PyMC model with weakly informative priors")
    print("="*60)

    with pm.Model() as model:
        
        alpha = pm.Normal('alpha', mu=0, sigma=10000)
        
        beta1 = pm.Normal('beta1', mu=0, sigma=1000)
        beta2 = pm.Normal('beta2', mu=0, sigma=1000)
        
        sigma = pm.HalfNormal('sigma', sigma=1000)
        
        mu = alpha + beta1 * x1 + beta2 * x2
        
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

    print("\nSampling completed!")
    print(pm.summary(trace, hdi_prob=0.95))

    # b)
    print("\n" + "="*60)
    print("Part b) 95% HDI estimates for β1 and β2")
    print("="*60)

    hdi_beta1 = az.hdi(trace, var_names=['beta1'], hdi_prob=0.95)
    hdi_beta2 = az.hdi(trace, var_names=['beta2'], hdi_prob=0.95)

    print(f"\n95% HDI for β1 (Speed coefficient): [{hdi_beta1['beta1'].values[0]:.4f}, {hdi_beta1['beta1'].values[1]:.4f}]")
    print(f"95% HDI for β2 (Log(HardDrive) coefficient): [{hdi_beta2['beta2'].values[0]:.4f}, {hdi_beta2['beta2'].values[1]:.4f}]")

    # c)
    print("\n" + "="*60)
    print("Part c) Usefulness of predictors")
    print("="*60)

    beta1_mean = trace.posterior['beta1'].mean().values
    beta2_mean = trace.posterior['beta2'].mean().values

    print(f"\nMean of β1: {beta1_mean:.4f}")
    print(f"Mean of β2: {beta2_mean:.4f}")

    # verifica daca hdi are zero
    beta1_includes_zero = (hdi_beta1['beta1'].values[0] <= 0 <= hdi_beta1['beta1'].values[1])
    beta2_includes_zero = (hdi_beta2['beta2'].values[0] <= 0 <= hdi_beta2['beta2'].values[1])

    print(f"\nDoes β1 95% HDI include 0? {beta1_includes_zero}")
    print(f"Does β2 95% HDI include 0? {beta2_includes_zero}")

    print("\nConclusion:")
    if not beta1_includes_zero:
        print("- Processor frequency (β1) is a USEFUL predictor: its HDI doesn't include 0.")
    else:
        print("- Processor frequency (β1) may NOT be a useful predictor: its HDI includes 0.")

    if not beta2_includes_zero:
        print("- Hard disk size (β2) is a USEFUL predictor: its HDI doesn't include 0.")
    else:
        print("- Hard disk size (β2) may NOT be a useful predictor: its HDI includes 0.")

    # d)
    print("\n" + "="*60)
    print("Part d) Expected price (μ) for 33 MHz and 540 MB HDD")
    print("="*60)

    x1_new = 33  # MHz
    x2_new = np.log(540)  

    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta1_samples = trace.posterior['beta1'].values.flatten()
    beta2_samples = trace.posterior['beta2'].values.flatten()

    # expeted price
    mu_new = alpha_samples + beta1_samples * x1_new + beta2_samples * x2_new

    # 90% HDI pentru expected price
    hdi_mu = az.hdi(mu_new, hdi_prob=0.90)

    print(f"\nExpected price μ for 33 MHz and 540 MB HDD:")
    print(f"Mean: ${mu_new.mean():.2f}")
    print(f"90% HDI: [${hdi_mu[0]:.2f}, ${hdi_mu[1]:.2f}]")

    # e)
    print("\n" + "="*60)
    print("Part e) Posterior predictive distribution for actual price")
    print("="*60)

    sigma_samples = trace.posterior['sigma'].values.flatten()

    # aici simulam modeul y ∼ N (μ, σ),
    y_pred = np.random.normal(mu_new, sigma_samples)

    # 90% HDI pentru predictie
    hdi_y_pred = az.hdi(y_pred, hdi_prob=0.90)

    print(f"\nPredicted sale price for 33 MHz and 540 MB HDD:")
    print(f"Mean: ${y_pred.mean():.2f}")
    print(f"90% HDI prediction interval: [${hdi_y_pred[0]:.2f}, ${hdi_y_pred[1]:.2f}]")

    print(f"\nNote: The prediction interval is wider than the expected price interval")
    print(f"because it accounts for both parameter uncertainty and observation noise.")

    # BONUS: Premium manufacturer effect
    # aici am sa fac un model nou care include variabila Premium si am sa vad daca are efect semnificativ asupra pretului
    print("\n" + "="*60)
    print("BONUS: Effect of Premium manufacturer")
    print("="*60)

    # am facut varaibila binara pentru premium
    premium = (df['Premium'] == 'yes').astype(int).values

    #aici fac modelul nou cu premium
    with pm.Model() as model_premium:
        
        alpha_p = pm.Normal('alpha', mu=0, sigma=10000)
        beta1_p = pm.Normal('beta1', mu=0, sigma=1000)
        beta2_p = pm.Normal('beta2', mu=0, sigma=1000)
        beta_premium = pm.Normal('beta_premium', mu=0, sigma=1000)  
        sigma_p = pm.HalfNormal('sigma', sigma=1000)
        
        mu_p = alpha_p + beta1_p * x1 + beta2_p * x2 + beta_premium * premium
        
        y_obs_p = pm.Normal('y_obs', mu=mu_p, sigma=sigma_p, observed=y)
        
        trace_premium = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

    print("\nModel with Premium manufacturer:")
    print(pm.summary(trace_premium, var_names=['beta_premium'], hdi_prob=0.95))

    hdi_premium = az.hdi(trace_premium, var_names=['beta_premium'], hdi_prob=0.95)
    beta_premium_mean = trace_premium.posterior['beta_premium'].mean().values

    print(f"\n95% HDI for β_premium: [{hdi_premium['beta_premium'].values[0]:.4f}, {hdi_premium['beta_premium'].values[1]:.4f}]")
    print(f"Mean of β_premium: {beta_premium_mean:.4f}")

    premium_includes_zero = (hdi_premium['beta_premium'].values[0] <= 0 <= hdi_premium['beta_premium'].values[1])

    print(f"\nDoes β_premium 95% HDI include 0? {premium_includes_zero}")

    if not premium_includes_zero:
        if beta_premium_mean > 0:
            print("\nConclusion: Premium manufacturers command a HIGHER price.")
        else:
            print("\nConclusion: Premium manufacturers have a LOWER price.")
    else:
        print("\nConclusion: Premium manufacturer status does NOT significantly affect price.")

    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    
    az.plot_posterior(trace, var_names=['beta1'], hdi_prob=0.95, ax=axes[0, 0])
    axes[0, 0].set_title('Posterior of β1 (Speed)')

    az.plot_posterior(trace, var_names=['beta2'], hdi_prob=0.95, ax=axes[0, 1])
    axes[0, 1].set_title('Posterior of β2 (Log(HardDrive))')

   
    axes[1, 0].hist(mu_new, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(hdi_mu[0], color='red', linestyle='--', label='90% HDI')
    axes[1, 0].axvline(hdi_mu[1], color='red', linestyle='--')
    axes[1, 0].set_xlabel('Expected Price ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Expected Price for 33 MHz, 540 MB')
    axes[1, 0].legend()

    
    axes[1, 1].hist(y_pred, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 1].axvline(hdi_y_pred[0], color='red', linestyle='--', label='90% HDI')
    axes[1, 1].axvline(hdi_y_pred[1], color='red', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Price ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Predicted Price for 33 MHz, 540 MB')
    axes[1, 1].legend()

    plt.tight_layout()  
    plt.savefig('lab11_results.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*60)
    print("Visualizations saved to 'lab11_results.png'")
    print("="*60)