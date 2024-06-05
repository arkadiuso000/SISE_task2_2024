import my_functions as mf

training_data, testing_data = mf.load_data()

# models with 1 hidden layer
h1_v1 = mf.create_model([1],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
h1_v2 = mf.create_model([1],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
h1_v3 = mf.create_model([1],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')

# models with 2 hidden layer
h2_v1 = mf.create_model([2,2],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
h2_v2 = mf.create_model([2,2],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
h2_v3 = mf.create_model([2,2],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')

# models with 3 hidden layer
h3_v1 = mf.create_model([3,3,3],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
h3_v2 = mf.create_model([3,3,3],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')
h3_v3 = mf.create_model([3,3,3],activation_function='tanh', activation_function_out='linear', weight_init_method='glorot_uniform')


# training models with 1 hidden layer
# training models with 2 hidden layer
# training models with 3 hidden layer