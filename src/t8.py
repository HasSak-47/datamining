import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from .utils import remove_outliers_iqr

def main():
    df = remove_outliers_iqr( pd.read_csv('typed_uanl.csv'), 'Sueldo Neto')
    df_by_date = df.groupby('Fecha')['Sueldo Neto'].mean().reset_index(name='avg_salary')
    
    # Convert dates to numeric values if they're not already
    if not pd.api.types.is_numeric_dtype(df_by_date['Fecha']):
        df_by_date['date_numeric'] = range(len(df_by_date))
        x = df_by_date['date_numeric']
    else:
        x = df_by_date['Fecha']
    
    X = sm.add_constant(x)  # Adds constant term for intercept
    model = sm.OLS(df_by_date['avg_salary'], X).fit()
    
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df_by_date['Fecha'], df_by_date['avg_salary'], 
                alpha=0.5, label='Actual Salaries')
    
    plt.plot(df_by_date['Fecha'], model.predict(X), 
             color='red', label='Regression Line')
    
    predictions = model.get_prediction(X).summary_frame()
    plt.fill_between(df_by_date['Fecha'],
                    predictions['mean_ci_lower'],
                    predictions['mean_ci_upper'],
                    color='orange', alpha=0.2, 
                    label='95% Confidence Interval')
    
    plt.xlabel('Date')
    plt.ylabel('Average Monthly Salary')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
