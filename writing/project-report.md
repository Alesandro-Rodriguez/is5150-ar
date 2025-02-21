# Premier League Causal Analysis


## Description

This project aims to analyze the causal impact of coaching changes on
team performances during the 2023-2024 Premier League Season. Using a
dataset of match outcomes and coaching history, the analysis leverages
regression techniques to infer causality.

------------------------------------------------------------------------

## The Story

This project investigates the impact of mid-season head coaching changes
on team performance in the 2023–2024 Premier League season. Coaching
changes are often made in response to underperformance, but it’s unclear
whether they lead to measurable improvements in outcomes like win
percentage or league ranking.

The goal of this project is to analyze whether coaching changes improve
team performance and evaluate the extent of their impact. The findings
will provide insights into whether these changes justify their cost and
strategic importance.

------------------------------------------------------------------------

## Ideal Dataset

The analysis uses the following data:

1.  **Match Outcomes**: Match dates, results, scores, and home/away
    teams.
2.  **Coaching Changes**: A table of head coaching changes with start
    and end dates.
3.  **Additional Variables**:
    - **Team Strength**: Metrics like squad value or past performance.
    - **Opponent Strength**: League ranking or historical performance of
      opposing teams.
    - **Match Context**: Home vs. away games and timing relative to the
      coaching change.

------------------------------------------------------------------------

## Factors Affecting Outcomes

Performance is influenced by several factors:

- **Strategic Leadership**: New coaches may bring different tactics or
  motivational strategies.
- **Team Dynamics**: Changes may disrupt or improve the team’s cohesion.
- **Timing**: Early-season changes allow time for recovery, while
  late-season changes are often last-resort measures.
- **Player Quality**: The adaptability and skill of the players affects
  results.
- **Opponent Strength**: Performance depends on the difficulty of the
  opposing teams.

------------------------------------------------------------------------

## Addressing Feedback

- **What do you mean by coaching changes?**  
  Coaching changes refer to replacements of head coaches during the
  season. This analysis focuses only on head coaching changes for
  clarity and feasibility.

- **How often do coaching changes occur during a season?**  
  In the 2023–2024 Premier League season, there were approximately 7–10
  mid-season head coaching changes.

- **What’s the business angle?**  
  Coaching changes impact financial metrics like sponsorships, ticket
  sales, and avoiding relegation, which has major monetary implications.

- **Have you found relevant data?**  
  Yes, the analysis uses a Kaggle dataset for 2023–2024 Premier League
  matches, supplemented with coaching change data manually compiled from
  public records.

- **What would the outcome be?**  
  Key outcomes include:

  - **Win Percentage**: Improvement in match win rates after a coaching
    change.
  - **Goal Differential**: Changes in scoring and defensive performance.
  - **League Ranking**: Shifts in table position after coaching changes.

- **Other Variables to Include?**

  - Team and opponent strength.
  - Match location (home vs. away).
  - Timing of coaching changes (early vs. late season).

------------------------------------------------------------------------

## Theory

Coaching changes are typically a response to poor results. This project
examines whether these changes improve team performance or if outcomes
are primarily driven by other factors, such as team quality or opponent
difficulty. By controlling for these variables, the analysis aims to
estimate the true impact of coaching changes.

------------------------------------------------------------------------

## DAG

<img src="figures/UpdatedDag.png" data-fig-align="center" />

------------------------------------------------------------------------

## Identification Strategy

### Where We Started

The objective of this project is to analyze the causal impact of
coaching changes on team performance in the Premier League, using
**Points** as the outcome variable. Points are calculated as **3 for a
win, 1 for a draw, and 0 for a loss**. Our initial focus was on
identifying key variables and their relationships, as represented in the
DAG, to develop a strategy for isolating the causal effect of coaching
changes.

### Where We Are Going

The goal of this milestone is to specify an identification strategy that
ensures: - Key confounding variables are included to satisfy the
**backdoor criterion**. - Irrelevant variables are excluded to
streamline the model. - Decisions are based on sound **causal
assumptions** supported by the DAG.

### Variables to Include Based on the DAG

- **Team Strength**: Captures the quality of the team, which influences
  both the likelihood of a coaching change and performance outcomes.
- **Opponent Strength**: Reflects the difficulty of the opposing team,
  which impacts match outcomes and adjusts for match context.
- **Match Location**: Accounts for the home or away setting, which
  influences team performance.
- **Date**: Controls for temporal trends and seasonality, such as
  early-season vs. late-season effects.

### Variables to Exclude Based on the DAG

- **Start_Date and End_Date**: These are part of the treatment
  definition (coaching change) and should not be conditioned on
  directly.
- **HomeTeam and AwayTeam**: Team identity is indirectly captured by
  **Team Strength**.
- **Points**: As the outcome variable, it should not be conditioned on.

------------------------------------------------------------------------

## Simulation

import numpy as np import polars as pl import seaborn as sns from
sklearn.linear_model import LinearRegression

np.random.seed(42)

# Define true parameter values

beta0 = 3  
beta1 = 2  
beta2 = 5  
beta3 = -3  
beta4 = 1  
n = 1000  
noise_sd = 2

# Simulate predictors

sim_data = ( pl.DataFrame({ “CoachChange”: np.random.choice(\[0, 1\],
size=n),  
“Opponent_Strength”: np.random.uniform(50, 100, size=n),  
“Match_Location”: np.random.choice(\[0, 1\], size=n)  
}) .with_columns(\[ (beta0 + beta1 \* pl.col(“CoachChange”) +
np.random.normal(0, noise_sd, size=n)).alias(“Team_Strength”) \])
.with_columns(\[ (beta2 \* pl.col(“Team_Strength”) + beta3 \*
pl.col(“Opponent_Strength”) + beta4 \* pl.col(“Match_Location”) +
np.random.normal(0, noise_sd, size=n)).alias(“Points”) \]) )

------------------------------------------------------------------------

# EDA

import pandas as pd import matplotlib.pyplot as plt import seaborn as
sns

file_path = “/Users/alesandro/Downloads/matches_corrected.csv” df =
pd.read_csv(file_path)

### BASIC DATA CHECKS

Display basic info print(“Dataset Info:”) df.info()

Display first few rows print(“ Rows:”) print(df.head())

Check for missing values print(“Values:”) print(df.isnull().sum())

Summary statistics print(“Statistics:”) print(df.describe())

### DISTRIBUTION PLOTS

Points Distribution plt.figure(figsize=(8, 5))
sns.histplot(df\[“Points”\], bins=3, kde=True) plt.title(“Distribution
of Points”) plt.xlabel(“Points”) plt.ylabel(“Frequency”) plt.show()

Team Strength Distribution plt.figure(figsize=(8, 5))
sns.histplot(df\[“Team_Strength”\], bins=20, kde=True)
plt.title(“Distribution of Team Strength”) plt.xlabel(“Squad Value
(£M)”) plt.ylabel(“Frequency”) plt.show()

Opponent Strength Distribution plt.figure(figsize=(8, 5))
sns.histplot(df\[“Opponent_Strength”\], bins=20, kde=True)
plt.title(“Distribution of Opponent Strength”) plt.xlabel(“Squad Value
(M)”) plt.ylabel(“Frequency”) plt.show()

### CORRELATION ANALYSIS

plt.figure(figsize=(10, 6)) sns.heatmap(df\[\[“Points”, “Team_Strength”,
“Opponent_Strength”, “CoachChange”\]\].corr(), annot=True,
cmap=“coolwarm”, fmt=“.2f”) plt.title(“Correlation Heatmap”) plt.show()

### CATEGORICAL ANALYSIS

Countplot for Coaching Changes plt.figure(figsize=(8, 5))
sns.countplot(x=“CoachChange”, data=df, palette=“Set2”)
plt.title(“Distribution of Coaching Changes”) plt.xlabel(“Coach Change
(0 = No, 1 = Yes)”) plt.ylabel(“Frequency”) plt.show()

Points by Change Timing plt.figure(figsize=(10, 5))
sns.boxplot(x=“Change_Timing”, y=“Points”, data=df, palette=“Set1”)
plt.title(“Points Distribution by Change Timing”) plt.xlabel(“Change
Timing”) plt.ylabel(“Points Earned”) plt.xticks(rotation=45) plt.show()
