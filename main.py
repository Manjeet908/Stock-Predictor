import load_dataset
import modelbuilding
import predict
import train_model
import visulize

# Load and preprocess
X_train, y_train, scaler, dataset_train, dataset_test = load_dataset.load_and_preprocess_data('Google_train_data.csv', 'Google_test_data.csv')

# Build model
model = modelbuilding.build_lstm_model((X_train.shape[1], 1))

# Train model
history = train_model.train_model(model, X_train, y_train)

# Predict
predicted_stock_price = predict.predict_stock_price(model, dataset_train, dataset_test, scaler)

# Visualize
actual_stock_price = dataset_test.iloc[:, 1:2].values
visulize.visualize_results(actual_stock_price, predicted_stock_price)