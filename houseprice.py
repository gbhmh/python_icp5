import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

house = pd.read_csv('train.csv', sep=',',usecols=(62,80))

# print(house.columns)
# print(house.info())
# print(house.head())

plt.scatter(house['GarageArea'],house['SalePrice'])
plt.title("Original Dataframe")
plt.ylabel("Sales Price")
plt.xlabel("Garage Area")
plt.show()

# Q1 = house.quantile(0.25)
# Q3 = house.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# house1 = house[~((house < (Q1 - 1000 * IQR)) |(house > (Q3 + 2.2 * IQR))).any(axis=1)]
# house1.shape
# plt.show()
#
z = np.abs(stats.zscore(house))
# print(z)
threshold = 2.2
print(np.where(z > 2.2))
house1 = house[(z < 2.2).all(axis=1)]
#
plt.scatter(house1['GarageArea'],house1['SalePrice'])
plt.title("Modified Dataframe")
plt.ylabel("Sales Price")
plt.xlabel("Garage Area")
plt.show()

# plt.subplot(1,2,1)
# plt.scatter(house['GarageArea'],house['SalePrice'], 'r--') # More on color options later
# plt.subplot(1,2,2)
# plt.scatter(house1['GarageArea'],house1['SalePrice'], 'g*-');
# plt.show()
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(house['GarageArea'],house['SalePrice'])
# ax.set_xlabel('GarageArea')
# ax.set_ylabel('SalePrice')
# plt.show()