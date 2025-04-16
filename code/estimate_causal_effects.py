import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("/Users/alesandro/Downloads/matches_corrected.csv")

# Convert 'Venue' to numeric (1 = Home, 0 = Away)
df["Venue"] = df["Venue"].map({"Home": 1, "Away": 0})

# Convert 'Date' to numeric
df["Date_numeric"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").astype(int) / 10**9

# One-hot encode Team (drop first to avoid multicollinearity)
team_dummies = pd.get_dummies(df["Team"], prefix="Team", drop_first=True)

# Combine all predictors: CoachChange, Date, Venue, Team dummies
X_raw = pd.concat([
    df[["CoachChange", "Date_numeric", "Venue"]],
    team_dummies
], axis=1)

# Standardize Date and Venue only
scaler = StandardScaler()
X_raw[["Date_numeric", "Venue"]] = scaler.fit_transform(X_raw[["Date_numeric", "Venue"]])

# Final arrays
X = X_raw.astype(float).to_numpy()
y = df["Points"].to_numpy()

# Bayesian model
with pm.Model() as coaching_model:
    X_data = pm.Data("X_data", X)
    y_data = pm.Data("y_data", y)

    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=5, shape=X.shape[1])
    sigma = pm.Exponential("sigma", lam=1)

    # Likelihood
    mu = alpha + pm.math.dot(X_data, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    # Sampling
    idata = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.95, cores=1)

# Posterior summary
summary = az.summary(idata, round_to=2)
print(summary)

# Trace plots
az.plot_trace(idata, combined=True)

