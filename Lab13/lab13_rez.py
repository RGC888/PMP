import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import os

az.style.use('arviz-darkgrid')

def load_data(filename="date.csv"):

    if os.path.exists(filename):
        print(f"Loading data from '{filename}'...")
        try:
            df = pd.read_csv(filename)
            if 'x' not in df.columns:
                df = pd.read_csv(filename, header=None, names=['x', 'y'])
            
            x = df['x'].values
            y = df['y'].values
            return x, y
        except Exception as e:
            print(f"Error reading file: {e}. Generating dummy data instead.")
    else:
        print(f"File '{filename}' not found. Generating dummy data instead.")
    return generate_dummy_data(n=20)

def generate_dummy_data(n=20, seed=42):
    #prob nu mai am nevoice de asta deoarece am csv-ul merge acm
    #le voi folosi in schimb cand generez date dummy pt ex 1.2
    np.random.seed(seed)
    x = np.linspace(-1.7, 1.7, n)
    y = 1.5*x - 1.0*x**2 + 0.5*x**3 + np.random.normal(0, 0.5, size=n)
    return x, y

def get_standardized_data(x, y, order):

    # creare de puteri [x^1, x^2, ..., x^order]
    x_1p = np.vstack([x**i for i in range(1, order+1)])
    
    # Standardizare caracteristici
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    
    # Standardizare target
    y_1s = (y - y.mean()) / y.std()
    
    return x_1s, y_1s, x_1p


def run_model_p(x_data_s, y_data_s, order, beta_sd):

    with pm.Model() as model_p:
       
        alpha = pm.Normal('alpha', mu=0, sigma=1)

        beta = pm.Normal('beta', mu=0, sigma=beta_sd, shape=order)
        
        epsilon = pm.HalfNormal('epsilon', 5)
        
        # liniar
        mu = alpha + pm.math.dot(beta, x_data_s)
        
        # Likelihood
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_data_s)
        
        idata = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=False)
        
        pm.compute_log_likelihood(idata)
        
    return idata, model_p


def plot_fit(idata, x_s, x_raw, y_raw, title, color):
    post = az.extract(idata)
    
    alpha_mean = post['alpha'].mean().item()
    beta_mean = post['beta'].mean(dim='sample').values
    
    y_pred_s = alpha_mean + np.dot(beta_mean, x_s)
    
    idx = np.argsort(x_raw)
    
    y_std_raw = (y_raw - y_raw.mean()) / y_raw.std()
    
    plt.plot(x_raw[idx], y_pred_s[idx], color=color, label=title, linewidth=2)


if __name__ == "__main__":
    
    # 1.1
    print("\n--- Exercise 1.1: Order=5 (date.csv) ---")
    
    x, y = load_data("date.csv")
    order = 5
    x_s, y_s, _ = get_standardized_data(x, y, order)
    
    plt.figure(figsize=(10, 6))
    y_plot_std = (y - y.mean()) / y.std()
    plt.scatter(x, y_plot_std, c='black', alpha=0.6, label='Data (Standardized)')
    
    # sd=10
    print("  > Fitting with sd=10...")
    idata_10, _ = run_model_p(x_s, y_s, order, beta_sd=10)
    plot_fit(idata_10, x_s, x, y, title="sd=10 (Default)", color='C1')
    
    # sd=100
    print("  > Fitting with sd=100...")
    idata_100, _ = run_model_p(x_s, y_s, order, beta_sd=100)
    plot_fit(idata_100, x_s, x, y, title="sd=100 (Flat/Overfit)", color='C2')
    
    # sd=[10, 0.1...]
    print("  > Fitting with Regularized Prior...")
    sigma_vec = np.array([10, 0.1, 0.1, 0.1, 0.1])
    idata_reg, _ = run_model_p(x_s, y_s, order, beta_sd=sigma_vec)
    plot_fit(idata_reg, x_s, x, y, title="sd=[10, 0.1...] (Regularized)", color='C3')
    
    plt.title(f"Exercise 1.1: Polynomial Fit (Order 5, N={len(x)})")
    plt.legend()
    plt.show()

    # 1.2    
    print("\n--- Exercise 1.2: Order=5 (Simulated N=500) ---")
    
    # genraare de 500 de pcte
    x_large, y_large = generate_dummy_data(n=500)
    x_s_large, y_s_large, _ = get_standardized_data(x_large, y_large, order=5)
    
    print("  > Fitting N=500 with Regularized Prior...")
    idata_large, _ = run_model_p(x_s_large, y_s_large, order=5, beta_sd=sigma_vec)
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_large, (y_large - y_large.mean())/y_large.std(), c='black', alpha=0.3, s=10, label='Data N=500')
    plot_fit(idata_large, x_s_large, x_large, y_large, title="N=500 Fit", color='C4')
    plt.title("Exercise 1.2: Large Data Overcomes Priors")
    plt.legend()
    plt.show()

    # 1.3
    print("\n--- Exercise 1.3: Model Comparison (date.csv) ---")
    
    compare_dict = {}
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_plot_std, c='black', alpha=0.6, label='Data')
    
    for ord_val in [1, 2, 3]:
        print(f"  > Fitting Order {ord_val}...")
        
        xs, ys, _ = get_standardized_data(x, y, order=ord_val)
        
        # sd=10
        trace, model = run_model_p(xs, ys, order=ord_val, beta_sd=10)
        
        compare_dict[f'Order {ord_val}'] = trace

        # plot
        plot_fit(trace, xs, x, y, title=f'Order {ord_val}', color=f'C{ord_val}')
        
    plt.title("Exercise 1.3: Linear vs Quadratic vs Cubic")
    plt.legend()
    plt.show()
    
    # WAIC Table
    print("\n--- WAIC Comparison Results ---")
    comp_df = az.compare(compare_dict, ic="waic", scale="deviance")
    print(comp_df)
    
    az.plot_compare(comp_df)
    plt.title("WAIC Comparison Plot")
    plt.show()