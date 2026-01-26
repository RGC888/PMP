import pandas as pd 
import pymc as pm
import numpy as np 
import arviz as az
import os

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'bike_daily.csv')
    csv_path = os.path.join(script_dir, '..', 'Examen', 'Prices.csv')
    df = pd.read_csv(data_path)
    print("Fișier încărcat cu succes!") 
    
    # Eliminăm rândurile cu valori NaN
    df = df.dropna()
    print(f"Dimensiune după eliminarea NaN: {len(df)} rânduri")
    
    y = df['rentals'].values
    
    x1 = df['temp_c'].values
    
    x2 = df['humidity'].values  
    x3 = df['wind_kph'].values
    x4 = df['is_holiday'].values
    
    season_mapping = {'spring': 0, 'summer': 1, 'fall': 2, 'winter': 3}
    x5 = df['season'].map(season_mapping).values
    
    # Standardizare date pentru a evita probleme numerice
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std
    
    x1_mean, x1_std = x1.mean(), x1.std()
    x1_scaled = (x1 - x1_mean) / x1_std
    
    x2_mean, x2_std = x2.mean(), x2.std()
    x2_scaled = (x2 - x2_mean) / x2_std
    
    x3_mean, x3_std = x3.mean(), x3.std()
    x3_scaled = (x3 - x3_mean) / x3_std
    
    x5_mean, x5_std = x5.mean(), x5.std()
    # Evităm diviziunea cu 0
    if x5_std > 0:
        x5_scaled = (x5 - x5_mean) / x5_std
    else:
        x5_scaled = x5 - x5_mean
    
    
    print(f"NaN în y_scaled: {np.isnan(y_scaled).any()}")
    print(f"NaN în x1_scaled: {np.isnan(x1_scaled).any()}")
    print(f"NaN în x2_scaled: {np.isnan(x2_scaled).any()}")
    print(f"NaN în x3_scaled: {np.isnan(x3_scaled).any()}")
    print(f"NaN în x5_scaled: {np.isnan(x5_scaled).any()}")
    
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        
        #pentru fiecare predictor definim un coeficient
       ## beta11 = pm.Normal("beta11", mu=0, sigma=1000)
        beta11 = 0 #dupa cum inteleg din enunt, la primul model nu folosesc niciun predictor pentru temp_c
        
        beta12 = pm.Normal("beta12", mu=0, sigma=10) 
        
        beta13 = pm.Normal("beta13", mu=0, sigma=10)
        
        beta14 = pm.Normal("beta14", mu=0, sigma=10)
        
        beta15 = pm.Normal("beta15", mu=0, sigma=10)
        
        sigma1 = pm.HalfNormal("sigma1", sigma=1)
    
        mu1 = alpha + beta12 * x2_scaled + beta13 * x3_scaled + beta14 * x4 + beta15 * x5_scaled 
        
        y_obs1 = pm.Normal('y_obs1', mu=mu1, sigma=sigma1, observed=y_scaled)
        
        trace1 = pm.sample(draws=1000, tune=1000, chains=3, target_accept=0.9, return_inferencedata=True, random_seed=42, initvals={'alpha': 0, 'beta12': 0, 'beta13': 0, 'beta14': 0, 'beta15': 0})
        
        pm.compute_log_likelihood(trace1)
        
        print("\nSampling completed!")
        
    with pm.Model() as model2:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        
        #pentru fiecare predictor definim un coeficient
        beta21 = pm.Normal("beta21", mu=0, sigma=10)
        
        beta22 = pm.Normal("beta22", mu=0, sigma=10) 
        
        beta23 = pm.Normal("beta23", mu=0, sigma=10)
        
        beta24 = pm.Normal("beta24", mu=0, sigma=10)
        
        beta25 = pm.Normal("beta25", mu=0, sigma=10)
        
        sigma2 = pm.HalfNormal("sigma2", sigma=1)
    
        mu2 = alpha + beta21*x1_scaled**2 + beta22 * x2_scaled + beta23 * x3_scaled + beta24 * x4 + beta25 * x5_scaled 
        
        y_obs2 = pm.Normal('y_obs2', mu=mu2, sigma=sigma2, observed=y_scaled)
        
        trace2 = pm.sample(draws=1000, tune=1000, chains=3, target_accept=0.9, return_inferencedata=True, random_seed=42, initvals={'alpha': 0, 'beta21': 0, 'beta22': 0, 'beta23': 0, 'beta24': 0, 'beta25': 0})
        
        print("\nSampling completed!")
    
    summary1 = pm.summary(trace1)
    summary2 = pm.summary(trace2)
    print(summary1)
    print(summary2)
    
    beta11_mean = 0
    beta12_mean = trace1.posterior['beta12'].mean().values
    beta13_mean = trace1.posterior['beta13'].mean().values
    beta14_mean = trace1.posterior['beta14'].mean().values  
    beta15_mean = trace1.posterior['beta15'].mean().values
    
    beta21_mean = trace2.posterior['beta21'].mean().values
    beta22_mean = trace2.posterior['beta22'].mean().values
    beta23_mean = trace2.posterior['beta23'].mean().values
    beta24_mean = trace2.posterior['beta24'].mean().values
    beta25_mean = trace2.posterior['beta25'].mean().values
    
    print(f"Model 1 Coeficienti:\n beta11: {beta11_mean}\n beta12: {beta12_mean}\n beta13: {beta13_mean}\n beta14: {beta14_mean}\n beta15: {beta15_mean}")
    print(f"Model 2 Coeficienti:\n beta21: {beta21_mean}\n beta22: {beta22_mean}\n beta23: {beta23_mean}\n beta24: {beta24_mean}\n beta25: {beta25_mean}")
    
    max1 = beta11_mean
    max2 = beta21_mean
    
    for var in ['beta12', 'beta13', 'beta14', 'beta15']:
        if abs(trace1.posterior[var].mean().values) > abs(max1):
            max1 = trace1.posterior[var].mean().values
            influence1 = var
    print(f"Model 1 Cel mai important predictor este {influence1} cu coeficientul {max1}")
    
    for var in ['beta21', 'beta22', 'beta23', 'beta24', 'beta25']:
        if abs(trace2.posterior[var].mean().values) > abs(max2):
            max2 = trace2.posterior[var].mean().values
            influence2 = var
    print(f"Model 2 Cel mai important predictor este {influence2} cu coeficientul {max2}")
     
    print("\n" + "="*50)
    print("COMPARAȚIE MODELE")
    print("="*50)
    
    waic1 = az.waic(trace1)
    waic2 = az.waic(trace2)
    
    print(f"\nWAIC Model 1: {waic1.waic:.2f}")
    print(f"WAIC Model 2: {waic2.waic:.2f}")
    
    loo1 = az.loo(trace1)
    loo2 = az.loo(trace2)
    
    print(f"\nLOO Model 1: {loo1.loo:.2f}")
    print(f"LOO Model 2: {loo2.loo:.2f}")
    
    compare_dict = {"Model 1": trace1, "Model 2": trace2}
    
    print("\n--- Comparație cu WAIC ---")
    comparison_waic = az.compare(compare_dict, ic="waic")
    print(comparison_waic)
    
    print("\n--- Comparație cu LOO ---")
    comparison_loo = az.compare(compare_dict, ic="loo")
    print(comparison_loo)
    
    print("\n" + "="*50)
    print("INTERPRETARE")
    print("="*50)
    print("Modelul cu valoarea WAIC/LOO mai mică este preferabil.")
    print("Rank 0 = cel mai bun model")
    g