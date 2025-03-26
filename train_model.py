def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return history
