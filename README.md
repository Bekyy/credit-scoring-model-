# credit-scoring-model-

## Overview
* The core objective of this project is to develop a Credit Scoring Model that can assess customer risk and predict their likelihood of defaulting on loans. 
* Credit scoring models are widely used in the financial industry to assign a quantitative measure to potential borrowers, helping institutions make informed lending decisions. By analyzing historical data on previous borrowers and their loan performance, these models offer insight into the creditworthiness of new applicants.

## Dataset overview
* The dataset for this challenge consists of transaction-level data, containing details about customer purchases, the platform's transaction processes, and fraud detection outcomes. Below is a detailed description of the key fields in the dataset:

- TransactionId: Unique transaction identifier on platform
- BatchId: Unique number assigned to a batch of transactions for processing
- AccountId: Unique number identifying the customer on platform
- SubscriptionId: Unique number identifying the customer subscription
- CustomerId: Unique identifier attached to Account
- CurrencyCode: Country currency
- CountryCode: Numerical geographical code of country
- ProviderId: Source provider of Item bought.
- ProductId: Item name being bought.
- ProductCategory: ProductIds are organised into these broader product categories.
- ChannelId: Identifies if customer used web,Android, IOS, pay later or checkout.
- Amount: Value of the transaction. Positive for debits from customer account and negative for credit into cus...
- Value: Absolute value of the amount
- TransactionStartTime: Transaction start time
- PricingStrategy: Category of Xente's pricing structure for merchants
- FraudResult: Fraud status of transaction 1 -yes or 0-No

These fields provide a diverse range of information that can be used for predictive modeling. Specifically, features like transaction amount, country, provider, product category, and platform channel can help identify patterns associated with fraudulent behavior.

## Objective
This project will focus on developing a robust credit scoring product that achieves the following key objectives:

1.	Define a proxy variable to categories customers as either high risk (bad) or low risk (good).
2.	Identify observable features that are strong predictors of customer default.
3.	Build a model that assigns a probability of default to new customers based on their characteristics.
4.	Translate risk probability into a credit score that reflects the customer's creditworthiness.
5.	Develop a model that predicts the optimal loan amount and duration for customers based on their risk profile.

## Major Tasks
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Default estimator and WoE binning
* Modelling
* Model Serving API Call

## Prerequests
* Python 3.x: Ensure Python is installed on your system.
* Virtual Environment: Recommended for managing project dependencies.

## Installation

1. Create a virtual environment:

**On macOS/Linux:**

```python -m venv venv```
```source venv/bin/activate```

**on windows:**
```python -m venv venv ```
```venv\Scripts\activate ```

2. Install dependencies:
``` pip install -r requirements.txt```


## Contribution

Contributions are welcome!

**License**

This project is licensed under the MIT License. See the LICENSE file for more details.
