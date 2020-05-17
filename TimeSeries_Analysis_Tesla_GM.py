# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:47:55 2020

@author: Rushi Shukla
"""

''' First, we import the required libraries and get some data. 
    Quandl automatically puts our data into a pandas dataframe, 
    the data structure of choice for data science. '''

# quandly for financial data
import quandl
# pandas for data manipulation
import pandas as pd

# Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Retrieve TSLA data from Quandl
quandl.ApiConfig.api_key = 'rFsSehe51RLzREtYhLfo'

tesla = quandl.get('WIKI/TSLA')
print(tesla.head(5))

# Retrieve the GM data from Quandl
gm = quandl.get('WIKI/GM')
gm.head(5)

print(gm.index)


'''Pandas dataframes can be easily plotted with matplotlib. 
    If any of the graphing code looks intimidating, don’t worry. 
    I also find matplotlib to be unintuitive and often copy and 
    paste examples from Stack Overflow or documentation to get 
    the graph I want.'''

# The adjusted close accounts for stock splits, so that is what we should graph
plt.plot(gm.index, gm['Adj. Close'])
plt.title('GM Stock Price')
plt.ylabel('Price ($)');
plt.show()

plt.plot(tesla.index, tesla['Adj. Close'], 'r')
plt.title('Tesla Stock Price')
plt.ylabel('Price ($)');
plt.show();

'''In order to compare the companies, we need to compute their market capitalization.
   Quandl does not provide this data, but we can figure out the market cap ourselves by multiplying 
   the average number of shares outstanding in each year times the share price.'''
   
   
'''Comparing the two companies on stock prices alone does not show which is more valuable 
   because the total value of a company (market capitalization) also depends on the number of 
   shares (Market cap= share price * number of shares). Quandl does not have number of shares data,
   but I was able to find average yearly stock shares for both companies with a quick Google search.
   Is is not exact, but will be accurate enough for our analysis. Sometimes we have to make do with
   imperfect data!'''
   
   
# Yearly average number of shares outstanding for Tesla and GM
tesla_shares = {2018: 168e6, 2017: 162e6, 2016: 144e6, 2015: 128e6, 2014: 125e6, 
                2013: 119e6, 2012: 107e6, 2011: 100e6, 2010: 51e6}

gm_shares = {2018: 1.42e9, 2017: 1.50e9, 2016: 1.54e9, 2015: 1.59e9, 2014: 1.61e9, 
                2013: 1.39e9, 2012: 1.57e9, 2011: 1.54e9, 2010: 1.50e9}


print(tesla.index.year)

''' To create a column of market cap in our dataframe,we use a few tricks with pandas, 
    such as moving the index to a column (reset_index) and simultaneously indexing and altering 
    values in the dataframe using ix. '''
    

# Create a year column
    
tesla['Year'] = tesla.index.year

# Take Dates from index and move to Date column 

tesla.reset_index(level=0, inplace = True)
tesla['cap'] = 0

# print(tesla.cap)
# Calculate market cap for all years

for i, year in enumerate(tesla['Year']):
    # Retrieve the shares for the year
    shares = tesla_shares.get(year)
#     print(shares)
    # Update the cap column to shares times the price
    tesla.loc[i, 'cap'] = shares * tesla.loc[i, 'Adj. Close']   
    
    
# print(shares) 
    

gm['Year'] = gm.index.year

# Take Dates from index and move to Date column 
gm.reset_index(level=0, inplace = True)
gm['cap'] = 0

# Calculate market cap for all years
for i, year in enumerate(gm['Year']):
    # Retrieve the shares for the year
    shares = gm_shares.get(year)
    
    # Update the cap column to shares times the price
    gm.loc[i, 'cap'] = shares * gm.loc[i, 'Adj. Close']    
    
# Merge the two datasets and rename the columns
cars = gm.merge(tesla, how='inner', on='Date')
print(cars)
cars.rename(columns={'cap_x': 'gm_cap', 'cap_y': 'tesla_cap'}, inplace=True)
print(cars) 


# Select only the relevant columns
cars = cars.loc[:, ['Date', 'gm_cap', 'tesla_cap']]

# Divide to get market cap in billions of dollars
cars['gm_cap'] = cars['gm_cap'] / 1e9
cars['tesla_cap'] = cars['tesla_cap'] / 1e9

cars.head()   

plt.figure(figsize=(10, 8))
plt.plot(cars['Date'], cars['gm_cap'], 'b-', label = 'GM')
plt.plot(cars['Date'], cars['tesla_cap'], 'r-', label = 'TESLA')
plt.xlabel('Date'); 
plt.ylabel('Market Cap (Billions $)'); 
plt.title('Market Cap of GM and Tesla')
plt.legend();

'''Tesla briefly surpassed GM in market cap in 2017. When did this occur?'''

import numpy as np

# Find the first and last time Tesla was valued higher than GM
first_date = cars.loc[np.min(list(np.where(cars['tesla_cap'] > cars['gm_cap'])[0])), 'Date']
last_date = cars.loc[np.max(list(np.where(cars['tesla_cap'] > cars['gm_cap'])[0])), 'Date']

print("Tesla was valued higher than GM from {} to {}.".format(first_date.date(), last_date.date()))

'''During Q2 2017, Tesla sold 22026 cars while GM sold 725000. In Q3 2017, Tesla sold 
   26137 cars and GM sold 808000. In all of 2017, Tesla sold 103084 cars and GM sold 3002237.
   That means GM was valued less than Tesla in a year during which it sold 29 times more cars 
   than Tesla! Interesting to say the least.'''
   
import fbprophet

# Prophet requires columns ds (Date) and y (value)
gm = gm.rename(columns={'Date': 'ds', 'cap': 'y'})
# Put market cap in billions
gm['y'] = gm['y'] / 1e9

# Make the prophet models and fit on the data
# changepoint_prior_scale can be changed to achieve a better fit
gm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05)
gm_prophet.fit(gm)

# Repeat for the tesla data
tesla =tesla.rename(columns={'Date': 'ds', 'cap': 'y'})
tesla['y'] = tesla['y'] / 1e9
tesla_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05, n_changepoints=10)
tesla_prophet.fit(tesla);   
   

# Make a future dataframe for 2 years
gm_forecast = gm_prophet.make_future_dataframe(periods=365 * 2, freq='D')
# Make predictions
gm_forecast = gm_prophet.predict(gm_forecast)

tesla_forecast = tesla_prophet.make_future_dataframe(periods=365*2, freq='D')
tesla_forecast = tesla_prophet.predict(tesla_forecast)    

gm_prophet.plot(gm_forecast, xlabel = 'Date', ylabel = 'Market Cap (billions $)')
plt.title('Market Cap of GM');

tesla_prophet.plot(tesla_forecast, xlabel = 'Date', ylabel = 'Market Cap (billions $)')
plt.title('Market Cap of Tesla');



    
    