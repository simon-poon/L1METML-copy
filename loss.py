import numpy as np

def custom_loss_wrapper(num_of_bins=20):
    bins = np.linspace(-500,500, num_of_bins-1)
    bin_center = bins - (bins[1] - bins[0])/2
    bin_center = np.append(bin_center, 500 + (bins[1] - bins[0])/2)
    bin_center = 
    
    def custom_loss(y_true, y_pred):
        print("-------------")
        print(y_true.shape)
        print(y_pred.shape) #(None, 2, 40)
        '''
        cutmoized loss function to improve the recoil response,
        by balancing the response above one and below one
        '''
        import tensorflow.keras.backend as K
        import tensorflow as tf

        px_truth = K.flatten(y_true[:, 0])
        py_truth = K.flatten(y_true[:, 1])
        px_pred = K.flatten(y_pred[:, 0])
        py_pred = K.flatten(y_pred[:, 1])

        #px_truth = y_true[:, 0, :]
        #py_truth = y_true[:, 1, :]
        #px_pred = y_pred[:, 0, :]
        #py_pred = y_pred[:, 1, :]

        pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

        loss = K.mean((px_truth - bin_center - px_pred)**2 / num_of_bins + (py_truth - bin_center - py_pred)**2 / num_of_bins)

        return loss

    return custom_loss
