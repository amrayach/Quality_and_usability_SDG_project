import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

desc = pd.read_excel('Company Info.xlsx', sheet_name='Company ListDescription')  # Read the description sheet
pred = pd.read_excel('Company Info.xlsx', sheet_name='Predictions')  # Read the prediction sheet

# Duplicate the Average Score from the prediction sheet to the description sheet
desc['Prediction Score'] = pred['Average Score']

goods = desc[desc['Company Type 0 - Goods / 1  - Services'] == 0]  # group the goods companies
serv = desc[desc['Company Type 0 - Goods / 1  - Services'] == 1]  # group the services companies

# Goods and Services Emission per Million Euros Average independently
meanG = goods['CO2 Emission per Million Euros [t CO2e / m EUR]'].mean()
meanS = serv['CO2 Emission per Million Euros [t CO2e / m EUR]'].mean()

# Goods and Services Emission per Million Euros Deviation independently
goods['Deviation'] = goods['CO2 Emission per Million Euros [t CO2e / m EUR]'] - meanG
serv['Deviation'] = serv['CO2 Emission per Million Euros [t CO2e / m EUR]'] - meanS

# maximum and minimum of the deviations to Normalize afterwards
deviation_minG = goods['Deviation'].min()
deviation_maxG = goods['Deviation'].max()
deviation_minS = serv['Deviation'].min()
deviation_maxS = serv['Deviation'].max()

# Normalizing Deviation and Scaling it to be between -1 and 1.
goods['Normalized Deviation'] = (goods['Deviation'] - deviation_minG) / (deviation_maxG - deviation_minG)  # Normalize
goods['Normalized Deviation'] = goods['Normalized Deviation'] * 2 - 1
serv['Normalized Deviation'] = (serv['Deviation'] - deviation_minS) / (deviation_maxS - deviation_minS)  # Normalize
serv['Normalized Deviation'] = serv['Normalized Deviation'] * 2 - 1

# Weights are parameters used to choose how much do we want to punish or reward the score from the previous phases
weightG = 0.5
weightS = 0.5

# Calculating an assigning the Final Score to each company based on the previous score (Average Score) and
# the normalized deviation. We punish and reward the Score with a factor of w*x^2, x being the normalized deviation
# and w the weight
goods['Final Score'] = goods['Prediction Score'] - weightG * np.multiply(goods['Normalized Deviation'],
                                                                         np.abs(goods['Normalized Deviation']))
serv['Final Score'] = serv['Prediction Score'] - weightS * np.multiply(serv['Normalized Deviation'],
                                                                       np.abs(serv['Normalized Deviation']))

# join the results
concatenated = pd.concat([goods, serv])
# reset indexes
concatenated.reset_index(drop=True, inplace=True)

# sort
concatenated.sort_values('Company Name', ascending=True)

# write the results
concatenated.to_excel('company_score_meta_data_prediction_results.xlsx', index=False)
