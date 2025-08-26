from sklearn.linear_model import LinearRegression
import numpy as np

def train_and_predict(X_train, y_train, X_test):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([2, 4, 6, 8, 10])

    
    X_test = np.array([[6]])

    
    y_pred = train_and_predict(X_train, y_train, X_test)

    
    print(f"Prediction for input {X_test.flatten()[0]} is {y_pred[0]:.2f}")

