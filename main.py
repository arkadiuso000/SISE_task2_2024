import my_functions as mf

training_data, testing_data = mf.load_data()
print(training_data.isna().sum())
print(testing_data.isna().sum())

# models with 1 hidden layer
h1_v1 = mf.create_model([1],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
# h1_v2 = mf.create_model([1],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
# h1_v3 = mf.create_model([1],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')

# models with 2 hidden layer
# h2_v1 = mf.create_model([2,2],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
# h2_v2 = mf.create_model([2,2],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
# h2_v3 = mf.create_model([2,2],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')

# models with 3 hidden layer
# h3_v1 = mf.create_model([3,3,3],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
# h3_v2 = mf.create_model([3,3,3],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
# h3_v3 = mf.create_model([3,3,3],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')


# training models with 1 hidden layer
h1 = mf.train_model(h1_v1,training_data, epochs=30)
t1 = mf.test_model(h1_v1,testing_data)
print(t1)
mf.plot(h1)
# training models with 2 hidden layer
# training models with 3 hidden layer