import numpy as np
import pandas as pd 
def predict_stock_price(model, dataset_train, dataset_test, scaler):

    # Prepare input for prediction
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, 300):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    return predicted_stock_price
