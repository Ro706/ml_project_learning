# Module section: 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#Function : 
def main():
    # Fetching data : 
    housing = datasets.fetch_california_housing() # This is fetching data from sklearn module -> california housing price
    # print(housing)
    X = housing.data
    y = housing.target

    # print(X[0])
    # print(y[0])
    
    # Data splitting : (training Data and Testing Data) 
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2 , random_state=42)
    '''
    test_size = 0.2 mean 20 80 rule: where 20% data is use for testing 80% data use for training
    random_state = 42 mean it control the randomness of the operations like :
     - Dataset splitting
     - shuffling 
     - algorithm initialization
    '''
    # Model Training : (linear Regression)
    model = LinearRegression()
    model.fit(X_train,y_train)
    '''
    Fitting the data into model by which in future it can predict the value
    About Linear Regression:
    '''
    
    #Prediction :
    y_pred = model.predict(X_test)
    #r2
    r2 = r2_score(y_test,y_pred)
    print(f"R2 Score: {r2 * 100} %",end="\n")
    #mae (mean absolute error)
    mae = mean_absolute_error(y_test,y_pred)
    print(f"MAE Score {mae * 100}%",end="\n")
    #mse (mean squared error)
    mse = mean_squared_error(y_test,y_pred)
    print(f"MSE Score {mse * 100}%",end="\n")
    #rmse (mean squared error with squared value false)
    rmse = np.sqrt(mse)
    print(f"RMSE Score {rmse * 100}%",end="\n")

    # Regression Ploting
    plt.scatter(y_test,y_pred,alpha=0.5,color="blue")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted values")
    plt.title("Actual vs Predicted Housing Prices")
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
    plt.show()
    
if __name__ == "__main__":
    main()
