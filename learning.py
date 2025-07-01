import numpy as np
from tensorflow.keras.datasets import mnist
from module_neuro_Maths import *
def softmaks(x):
    temp = np.exp(x)
    x /= np.sum(temp)
    return x



alpha = 0.0005
ind = 0
(x_train, y_train), (x_test, y_test) = mnist.load_data()
pixel_per_image = 28 * 28
hidden_size = 100
nums_labels = 10

itterations = 10000

def l():
    weights_0_1 = 2 * np.random.random((pixel_per_image, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, nums_labels)) - 1
    for i in range(itterations):
        sum_error = 0
        sum_error_test = 0
        for j in range(1000):
            # test (НЕ ОБУЧАТЬ)
            inp = transform_mat_in_vec(x_test[j])
            inp = vec_ele_div(inp, 256)
            layer_0 = np.array([inp])
            layer_1 = relu(layer_0.dot(weights_0_1))
            layer_2 = layer_1.dot(weights_1_2)


            goal_preds = paste_num_in_vec(y_test[j], 10)
            layer_2_delta = layer_2 - np.array([goal_preds])

            error = layer_2_delta ** 2
            sum_error_test += sum(error[0])
            # train
            inp = transform_mat_in_vec(x_train[j])
            inp = vec_ele_div(inp, 256)
            layer_0 = np.array([inp])
            layer_1 = np.tanh(layer_0.dot(weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask
            layer_2 = softmaks(layer_1.dot(weights_1_2))

            goal_preds = paste_num_in_vec(y_train[j], 10)
            layer_2_delta = layer_2 - np.array([goal_preds])
            layer_1_delta = layer_2.dot(weights_1_2.T) * (1 - (layer_1 ** 2))

            layer_1_delta *= dropout_mask

            wd_1_2 = layer_1.T.dot(layer_2_delta)
            wd_0_1 = layer_0.T.dot(layer_1_delta)
            weights_1_2 -= wd_1_2 * alpha
            weights_0_1 -= wd_0_1 * alpha

            error = layer_2_delta ** 2
            sum_error += sum(error[0])
        print(f"I:{i} Train-Err:{sum_error/10000} Test-Err:{sum_error_test/10000}")
        np.savetxt(f"weights_v2_2/weights_0_1({i}).txt", weights_0_1)
        np.savetxt(f"weights_v2_2/weights_1_2({i}).txt", weights_1_2)

z = 150
weights_0_1 = np.loadtxt(f"weights_v2_2/weights_0_1({z}).txt")
weights_1_2 = np.loadtxt(f"weights_v2_2/weights_1_2({z}).txt")