import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'date_promovare_examen.csv')
    
    try:
        df = pd.read_csv(csv_path)
        print("Fișier încărcat cu succes!")
    except FileNotFoundError:
        print(f"Eroare: Nu găsesc fișierul csv la: {csv_path}")
        exit()


    counts = df['Promovare'].value_counts()
    print("\n--- Distribuția Claselor ---")
    print(counts)


    if abs(counts[0] - counts[1]) < len(df) * 0.1:
        print("Concluzie: Datele sunt balansate.")
    else:
        print("Concluzie: Datele sunt nebalansate.")


    X_studiu = df['Ore_Studiu'].values
    X_somn = df['Ore_Somn'].values
    y_promovat = df['Promovare'].values
  
    studiu_mean, studiu_std = X_studiu.mean(), X_studiu.std()
    somn_mean, somn_std = X_somn.mean(), X_somn.std()

    X_studiu_s = (X_studiu - studiu_mean) / studiu_std
    X_somn_s = (X_somn - somn_mean) / somn_std

    print("\nÎncepe eșantionarea (sampling)... Așteaptă câteva secunde.")
    with pm.Model() as model_logistic:
        #priorii
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_studiu = pm.Normal('beta_studiu', mu=0, sigma=10)
        beta_somn = pm.Normal('beta_somn', mu=0, sigma=10)
        
        mu = alpha + beta_studiu * X_studiu_s + beta_somn * X_somn_s
        
        p = pm.math.sigmoid(mu)
        
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y_promovat)
        
        idata = pm.sample(1000, tune=1000, return_inferencedata=True)

    summary = az.summary(idata)
    print("\n--- Rezumat Parametri ---")
    print(summary)

    posterior_means = summary['mean']
    alpha_m = posterior_means['alpha']
    beta1_m = posterior_means['beta_studiu']
    beta2_m = posterior_means['beta_somn']

    #  x2 = -(alpha + b1*x1)/b2
    x_plot_s = np.linspace(X_studiu_s.min(), X_studiu_s.max(), 100)
    y_plot_s = -(alpha_m + beta1_m * x_plot_s) / beta2_m

    # ma reintorc la val reale
    x_plot_real = x_plot_s * studiu_std + studiu_mean
    y_plot_real = y_plot_s * somn_std + somn_mean

    plt.figure(figsize=(10, 6))
    plt.scatter(df[df['Promovare']==0]['Ore_Studiu'], df[df['Promovare']==0]['Ore_Somn'], 
                color='red', label='Picat (0)', alpha=0.6)
    plt.scatter(df[df['Promovare']==1]['Ore_Studiu'], df[df['Promovare']==1]['Ore_Somn'], 
                color='blue', label='Promovat (1)', alpha=0.6)
    plt.plot(x_plot_real, y_plot_real, color='black', linewidth=3, label='Granița de decizie')
    plt.xlabel('Ore Studiu / Săptămână')
    plt.ylabel('Ore Somn / Zi')
    plt.title('Regresie Logistică Bayesiană: Granița de Decizie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    z_pred = alpha_m + beta1_m * X_studiu_s + beta2_m * X_somn_s
    p_pred = 1 / (1 + np.exp(-z_pred))
    
    y_pred = (p_pred > 0.5).astype(int)

    accuracy = np.mean(y_pred == y_promovat)


    TP = np.sum((y_pred == 1) & (y_promovat == 1))

    TN = np.sum((y_pred == 0) & (y_promovat == 0))

    FP = np.sum((y_pred == 1) & (y_promovat == 0))

    FN = np.sum((y_pred == 0) & (y_promovat == 1))

    print("\n--- Separabilitatea Datelor ---")
    print(f"Acuratețe model: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print("Matrice de confuzie:")
    print(f"[[{TN} (TN), {FP} (FP)]")
    print( f" [{FN} (FN), {TP} (TP)]]")

    if accuracy > 0.85:
        print("Concluzie: Datele sunt BINE SEPARATE.")
    else:
        print("Concluzie: Datele nu sunt perfect separate (există overlap).")


    print("\n--- Concluzie Influență ---")
    print(f"Coeficient Studiu (Standardizat): {beta1_m:.3f}")
    print(f"Coeficient Somn (Standardizat): {beta2_m:.3f}")

    if abs(beta1_m) > abs(beta2_m):
        print("-> Orele de STUDIU influențează mai mult promovabilitatea.")
    else:
        print("-> Orele de SOMN influențează mai mult promovabilitatea.")