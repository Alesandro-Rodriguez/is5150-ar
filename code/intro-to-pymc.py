import numpy as np
import polars as pl
import pymc as pm
import arviz as az

np.random.seed(42)

#set parameters
beta0 = 3
beta1 = 7
sigma = 3
n = 100

#simulate data
x = np.random.uniform(0,7,size=n)
y = beta0 + beta1 * x + np.random.normal(0,sigma,size=n)

#create a model object
basic_model = pm.Model()

#specify the model
with basic_model:
    #prior.
    beta = pm.Normal('beta', mu = 0, sigma = 10, shape = 2)
    # beta0 = pm.Normal('beta0', mu = 0, sigma = 10)
    # beta1 = pm.Normal('beta1', mu = 0, sigma = 10)
    sigma = pm.HalfNormal('sigma', sigma = 1)

    #likelihood
    mu = beta[0] + beta[1] * x
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = y) #p(x|theta)

#create an inference object
with basic_model:
    #draw 100 posterior samples
    idata = pm.sample()

#have we recovered parameters
az.summary(idata, round_to = 2)

#visualize mariginal posteriors
az.plot_trace(idata, combined = True)



#import foxes data
foxes = pl.read_csv('/Users/alesandro/Downloads/foxes.csv')
#separate preidctors and the outcome
X = foxes.select(pl.col(['avgfood', 'groupsize'])).to_numpy()
y= foxes.select(pl.col('weight')).to_numpy().flatten()

with pm.Model() as foxes_model:
    #data
    X_data = pm.Data('X_data', X)
    y_data = pm.Data('y_data', y)

    #priors
    alpha = pm.Normal('alpha', mu = 0, sigma =.2)
    beta = pm.Normal('beta', mu = 0, sigma =.5, shape = 2)
    sigma = pm.Exponential('sigma', lam = 1)

    #likelihood
    mu = alpha + X_data @ beta
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = y_data)

#sample.
with foxes_model:
    draws = pm.sample


