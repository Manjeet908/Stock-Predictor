import matplotlib.pyplot as plt
def visualize_results(actual_stock_price, predicted_stock_price):

    plt.figure(figsize=(10,5))
    plt.plot(actual_stock_price, color='red', label='Actual Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig("plot.jpg")