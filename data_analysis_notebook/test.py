import numpy as np


if __name__ == "__main__":
    X = np.array([[1, 2, 3], [5, 2, 1], [8, 3, 1], [5, 2, 1]])

    match = [5, 2, 1]


    a = [x for x in a if x != [1,1]]

    
    # print(X.shape)
    # exit()

 #    # Get sum squares err
 #    X_hat = np.sum(np.mean(X_test)) / n_samples
 #    # X_means = [np.mean(x_mat) for x_mat in X_test]
 #    reconstruct_X_test = model.inverse_transform(W_test)

 #    SS_err = np.sum(X_test - reconstruct_X_test)**2
 #    SS_tot = np.sum(X_test - X_hat)**2

 #    fuv = SS_err / SS_tot
 #    print(fuv)

 #    test_error = _beta_divergence(X_test, W_test, model.components_, 'frobenius', square_root=False)
 #    k_error_dict[i].append(test_error)

 #    print("rep: ", rep, " k: ", i, "mean test error: ", np.mean(k_error_dict[i]))
