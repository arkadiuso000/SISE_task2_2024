from my_functions import *

training_data, testing_data = load_data()
# one hidden layer
# m_1_v1 = NeuralNetworkModel(hidden_layers=[90], epochs=200,learning_rate=0.01, optimizer='sgd',momentum=0.7, id=11)
# m_1_v2 = NeuralNetworkModel(hidden_layers=[1], epochs=30,learning_rate=0.01, optimizer='adam', id=12)
best_1 = NeuralNetworkModel(hidden_layers=[128], epochs=100, learning_rate=0.01, optimizer='adam',activation_function='relu', id=1)
# two hidden layers
# m_2_v1 = NeuralNetworkModel(hidden_layers=[90,110], epochs=200,learning_rate=0.01, optimizer='sgd',momentum=0.7, id=21)
# m_2_v2 = NeuralNetworkModel(hidden_layers=[1,2], epochs=30,learning_rate=0.01, optimizer='adam', id=22)
best_2 = NeuralNetworkModel(hidden_layers=[128,128], epochs=100, learning_rate=0.01, optimizer='adam',activation_function='relu', id=2)
# three hidden layers
# m_3_v1 = NeuralNetworkModel(hidden_layers=[90,110, 115], epochs=200,learning_rate=0.01, optimizer='sgd',momentum=0.7, id=31)
# m_3_v2 = NeuralNetworkModel(hidden_layers=[1,2,3], epochs=30,learning_rate=0.01, optimizer='adam', id=32)
best_3 = NeuralNetworkModel(hidden_layers=[128,128,128], epochs=100, learning_rate=0.01, optimizer='adam',activation_function='relu', id=3)

# history one hidden layer
# h_m_1_v1 = m_1_v1.train(training_data,testing_data)
# h_m_1_v2 = m_1_v2.train(training_data,testing_data)
# history two hidden layers
# h_m_2_v1 = m_2_v1.train(training_data,testing_data)
# h_m_2_v2 = m_2_v2.train(training_data,testing_data)
# history three hidden layers
# h_m_3_v1 = m_3_v1.train(training_data,testing_data)
# h_m_3_v2 = m_3_v2.train(training_data,testing_data)

# the best history
h_best_1 = best_1.train(training_data,testing_data)
h_best_2 = best_2.train(training_data,testing_data)
h_best_3 = best_3.train(training_data,testing_data)

# mse and predictions of one hidden layer
# mse_m_1_v1, pred_m_1_v1 = m_1_v1.test(testing_data)
# mse_m_1_v2, pred_m_1_v2 = m_1_v2.test(testing_data)
# mse and predictions of two hidden layers
# mse_m_2_v1, pred_m_2_v1 = m_2_v1.test(testing_data)
# mse_m_2_v2, pred_m_2_v2 = m_2_v2.test(testing_data)
# mse and predictions of three hidden layers
# mse_m_3_v1, pred_m_3_v1 = m_3_v1.test(testing_data)
# mse_m_3_v2, pred_m_3_v2 = m_3_v2.test(testing_data)
# mse and predictions of the best
mse_best_1, pred_best_1 = best_1.test(testing_data)
mse_best_2, pred_best_2 = best_2.test(testing_data)
mse_best_3, pred_best_3 = best_3.test(testing_data)


# worse plots
# plot_1([h_m_1_v2, h_m_2_v2, h_m_3_v2])
# plot_2([h_m_1_v2, h_m_2_v2, h_m_3_v2], testing_data)
# plot_3([m_1_v2,m_2_v2,m_3_v2],testing_data)
# plot_4([m_1_v2,m_2_v2,m_3_v2],testing_data)

# best plots
plot_1([h_best_1,h_best_2,h_best_3])
plot_2([h_best_1,h_best_2,h_best_3], testing_data)
plot_3([best_1, best_2,best_3],testing_data)
plot_4([best_1, best_2,best_3],testing_data)