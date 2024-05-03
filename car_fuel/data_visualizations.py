import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('data_CO2_emission.csv')
    
    print(data)

    plt.scatter(data['Cylinders'], data['Fuel Consumption City (L/100km)'])
    plt.title('Fuel consumption city in regards to number of cylinders')
    plt.xlabel('Number of cylinders')
    plt.ylabel('Fuel consumption city (L/100km)')
    plt.show()

    plt.scatter(data['Engine Size (L)'], data['Fuel Consumption City (L/100km)'])
    plt.title('Fuel consumption city in regards to engine size')
    plt.xlabel('Engine size (L)')
    plt.ylabel('Fuel consumption city (L/100km)')
    plt.show()

    data.boxplot(column='Fuel Consumption City (L/100km)', by='Fuel Type')
    plt.show()

    plt.scatter(data['Fuel Consumption City (L/100km)'], data['Fuel Consumption Hwy (L/100km)'], c='red')
    plt.title('Fuel consumption in city vs highway')
    plt.xlabel('Fuel consumption city (L/100km)')
    plt.ylabel('Fuel consumption highway (L/100km)')
    plt.show()
    
    data['Is Manual'] = data['Transmission'].apply(lambda x: True if x.startswith('M') else False)
    print(data['Is Manual'])
    print(data[data['Is Manual'] == True])

if __name__ == "__main__":
    main()
