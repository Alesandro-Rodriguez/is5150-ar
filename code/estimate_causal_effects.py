import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# Load dataset
df = pd.read_csv("/Users/alesandro/Downloads/matches_corrected.csv")

# Convert 'Venue' to numeric (1 for Home, 0 for Away)
df["Venue"] = df["Venue"].map({"Home": 1, "Away": 0})

# Prepare data (predictors and outcome)
X = df[["CoachChange", "Team_Strength", "Opponent_Strength", "Venue"]].to_numpy()
y = df["Points"].to_numpy()

# Bayesian model
with pm.Model() as coaching_model:
    # Data containers
    X_data = pm.Data("X_data", X)
    y_data = pm.Data("y_data", y)

    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=1)  # Intercept
    beta = pm.Normal("beta", mu=0, sigma=1, shape=4)  # One for each predictor
    sigma = pm.Exponential("sigma", lam=1)  # Standard deviation of error term

    # Likelihood
    mu = alpha + pm.math.dot(X_data, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    # Sample from the posterior
    idata = pm.sample(1000, tune=1000, return_inferencedata=True)

# Summarize posterior estimates
summary = az.summary(idata, round_to=2)
print(summary)

# Visualize posterior distributions
az.plot_trace(idata, combined=True)
