import numpy as np
def train_data_gen(num):
    """ Generate num training datapoints
        X is an 8D vector where the first 6D are frame numbers and the rest 2 are starting y and z position.
        y is an 10D vector, indicating the (y, z) in the next 5 frame steps.
    """
    
    obstacle = False
    # For wrapping, training data should be X: (end_init_x, end_init_y, trashcan_init_x, trashcan_init_y)
    #                                       y: ((end_x, end_y) * 29) T == 29
    if (obstacle):
        train_x = np.zeros((8, num))
        train_y = np.zeros((10, num))
        for i in range(num):
            
            train_x[:, i] = np.array([0, 20, 25, 30, 35, 50, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)])
            d1 = np.random.uniform(-0.1, 0.1)
            d2 = np.random.uniform(-0.1, 0.1)
            d3 = np.random.uniform(-0.1, 0.1)
            d4 = np.random.uniform(-0.1, 0.1)
            d5 = np.random.uniform(-0.05, 0.2)
            d6 = np.random.uniform(-0.05, 0.2)
            d7 = np.random.uniform(-0.05, 0.2)
            d8 = np.random.uniform(-0.05, 0.2)
            train_y[:, i] = np.array([0 + d5, 5 + d1, 0 + d6, 5.6 + d2, -2 + d7, 6 + d3, -3 + d8, 4.5 + d4, -3, 0])
        return train_x, train_y
    else:
        train_x = np.zeros((4, num))
        train_y = np.zeros((56, num))
        for i in range(num):
            dx = np.random.uniform(-1, 1)
            dy = np.random.uniform(-1, 1)
            train_x[:, i] = np.array([-13.225 + dx, 0 + dy, -10 + dx, -2.7 + dy])
            train_y[:, i] =              np.array([-13.225, -3.5,
                                            -8, -4,
                                            -4, -4,
                                            0.54, -4,
                                            0.62, -1.58, 
                                            -1.18, 0.78, 
                                            -4.52, -0.94,
                                            -8.24, -0.94,
                                            -11.34, -0.94,
                                            -14.66, -0.94,
                                            -18.42, -0.94,
                                            -15.72, -5.88,
                                            -10.8, -7.8, 
                                            -10.8, -12,
                                            -10.8, -16.34,
                                            -6.84, -14.42,
                                            -2.6, -6.14, 
                                            -0.56, -3.88,
                                            1, -3.88,
                                            -3.98, 0,
                                            -9.52, -1.22,
                                            -15.12, -1.4,
                                            -16.3, -1.4, 
                                            -17.4, -1.4, 
                                            -11.38, -8.78,
                                            -7.02, -3.98,
                                            -8.8, -0.94, 
                                            -8.8, 0])

            for j in range(len(train_y[:, i])):
                if j % 2 == 0:
                    train_y[:, i][j] += dx
                else:
                    train_y[:, i][j] += dy
        return train_x, train_y



if __name__ == "__main__":
    train_x, train_y = train_data_gen(200)
    print(train_x[:, 20])
    print(train_y[:, 20])    
    np.save('train_x_wrap.npy', train_x)
    np.save('train_y_wrap.npy', train_y)