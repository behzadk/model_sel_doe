import numpy as np


if __name__ == "__main__":
    X = np.array([[1, 2, 3], [5, 2, 1]])
    X_reg_match = np.array([[1, 1, 3], [5, 2, 1]])

    X = X.reshape(1, -1)
    X_reg_match = X_reg_match.reshape(1, -1)

    X_mean = np.mean(X)
    
    SS_err = np.sum((X - X_reg_match)**2)
    SS_tot = np.sum((X - X_mean)**2)
    
    print(X - X_mean)

    fuv = SS_err / SS_tot

    print(fuv)

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
