import numpy as np
import matplotlib.pyplot as plt

p_B = 0.01            
sensitivity = 0.95    
specificity = 0.90    

# a)
p_pos_given_notB = 1 - specificity 


numerator = sensitivity * p_B
denominator = sensitivity * p_B + p_pos_given_notB * (1 - p_B)
posterior = numerator / denominator

print("=== (a) Probability that a person actually has the disease given a positive test ===")
print(f"P(B | Positive) = {posterior:.6f}  ≈ {posterior*100:.2f}%\n")

print("Explanation:")
print("Even though the test is accurate, the disease is rare (1% prevalence).")
print("Most positive tests are actually false positives, giving only about an 8.8% chance "
      "that a person truly has the disease.\n")

# b)
s = sensitivity
p = p_B
required_specificity = 1 - (s * p) / (1 - p)

print("=== (b) Minimum specificity for P(B | Positive) = 0.5 ===")
print(f"Required specificity = {required_specificity:.6f}  ≈ {required_specificity*100:.2f}%\n")

# verificam
p_pos_given_notB_req = 1 - required_specificity
num_req = s * p
den_req = s * p + p_pos_given_notB_req * (1 - p)
posterior_req = num_req / den_req

print(f"Verification: P(B | Positive) = {posterior_req:.3f} (should be ≈ 0.5)\n")
