from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import max_error
from scipy.sparse import vstack

def main():
    data = pd.read_csv('data_CO2_emission.csv')

    input_variables = ['Fuel Type', 'Transmission', 'Engine Size (L)', 'Cylinders', 'Vehicle Class']
    output_variables = ['Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']

    X = data[input_variables]
    y = data[output_variables]
    
    one = OneHotEncoder()
    X_encoded = one.fit_transform(X[['Fuel Type', 'Transmission', 'Vehicle Class']])
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2)

    linear_regression_model = lm.LinearRegression()
    linear_regression_model.fit(X_train, y_train)

    y_test_p = linear_regression_model.predict(X_test)
    
    for column in output_variables:
        print('====================')
        index = output_variables.index(column)
        ME = max_error(y_test[column], y_test_p[:, index])
        print(f'Max error for {column}: {ME}')
        error = np.abs(y_test[column], y_test_p[:, index])
        max_error_id = np.argmax(error)
        max_error_model = data.iloc[max_error_id, 1]
        print(f'Model with the highest error: {max_error_model}')
        print('====================')

    cylinders = int(input('Enter the number of cylinders: '))
    engine_size = float(input('Enter engine size in liters: '))
    number_of_gears = int(input('Enter the number of gears: '))
    is_auto = bool(input('Is Automatic? (Enter True or False): '))
    transmission = ''
    if is_auto:
        transmission = f"A{number_of_gears}"
    else:
        transmission = f"M{number_of_gears}"
    vehicle_class = input('Enter a vehicle class: ')
    fuel_type = input('Enter fuel type: ')

    entered_data = {
        'Cylinders': [cylinders],
        'Engine Size (L)': [engine_size],
        'Transmission': [transmission],
        'Vehicle Class': [vehicle_class],
        'Fuel Type': [fuel_type],
    }
    
    entered_data = pd.DataFrame(entered_data)
    entered_data_sparse = one.transform(entered_data[['Fuel Type', 'Transmission', 'Vehicle Class']])
    X_encoded = vstack([X_encoded, entered_data_sparse])
    predicted_data = linear_regression_model.predict(X_encoded[-1].reshape(1, -1))
    print(f'Fuel Consumption City (L/100km): {predicted_data[0][0]}')
    print(f'Fuel Consumption Highway (L/100km): {predicted_data[0][1]}')
    print(f'Fuel Consumption Combined (L/100km): {predicted_data[0][2]}')

if __name__ == '__main__':
    main()
