from my_functions import *

training_data, testing_data = load_data()

# Tworzenie instancji modelu
model = NeuralNetworkModel(hidden_layers=[1], epochs=30, learning_rate=0.01, optimizer='adam', momentum=0.9)

# Trenowanie modelu
history = model.train(training_data, testing_data)

# Testowanie modelu
mse = model.test(testing_data)
print(f"MSE on test data: {mse}")

# Wizualizacja wyników
model.plot(history)
plot_cdf([model], testing_data)