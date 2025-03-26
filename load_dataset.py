def load_and_preprocess_data(train_file, test_file):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Load datasets
    dataset_train = pd.read_csv(train_file)
    dataset_test = pd.read_csv(test_file)

    # Extract 'Open' price
    training_set = dataset_train.iloc[:, 1:2].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_set = scaler.fit_transform(training_set)

    # Prepare training data
    X_train, y_train = [], []
    for i in range(60, len(scaled_training_set)):
        X_train.append(scaled_training_set[i-60:i, 0])
        y_train.append(scaled_training_set[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler, dataset_train, dataset_test
