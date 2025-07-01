def n(j):

    inp = transform_mat_in_vec(x_test[j])
    inp = vec_ele_div(inp, 256)
    layer_0 = np.array([inp])
    layer_1 = relu(layer_0.dot(weights_0_1))
    layer_2 = layer_1.dot(weights_1_2)

    goal_preds = paste_num_in_vec(y_test[j], 10)
    layer_2_delta = layer_2 - np.array([goal_preds])

    error = layer_2_delta ** 2
    sum_error_test = sum(error[0])
    print(f"target: {y_test[j]} pred: {layer_2[0]} er: {sum_error_test}")

for i in range(30):
    n(i)