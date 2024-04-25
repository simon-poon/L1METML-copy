import numpy as np

def custom_loss_wrapper(num_of_bins=20):
    bins = np.linspace(-500,500, num_of_bins-1)
    bin_center = bins - (bins[1] - bins[0])/2
    bin_center = np.append(bin_center, 500 + (bins[1] - bins[0])/2) 


    def custom_loss(y_true, y_pred):
        '''
        cutmoized loss function to improve the recoil response,
        by balancing the response above one and below one
        '''
        import tensorflow.keras.backend as K
        import tensorflow as tf
        bin_center_tf = tf.convert_to_tensor(bin_center, dtype=tf.float32, dtype_hint=None)
        bin_center_tf = tf.expand_dims(bin_center_tf,axis=0)
        bin_center_tf = tf.expand_dims(bin_center_tf,axis=0)
        for_broadcast = y_pred[:, 0:1, :] * 0
        bin_center_tf = for_broadcast + bin_center_tf
        bin_center_tf = K.flatten(bin_center_tf)
        
        #if y_true.shape[0] == None:
        #    m = np.zeros([256,1,num_of_bins])
        #else:
        #    m = np.zeros([y_true.shape[0],1,num_of_bins])
        #
        #bin_center_arr = m + bin_center[None, None, :]
        #bin_center_tf = tf.convert_to_tensor(bin_center_arr, dtype=tf.float32, dtype_hint=None)
        #bin_center_tf = K.flatten(bin_center_tf)

        px_truth = K.flatten(y_true[:, 0:1, :])
        py_truth = K.flatten(y_true[:, 1:2, :])
        px_pred = K.flatten(y_pred[:, 0:1, :])
        py_pred = K.flatten(y_pred[:, 1:2, :])

        pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

        loss = K.mean((px_truth - bin_center_tf - px_pred)**2 / num_of_bins + (py_truth - bin_center_tf - py_pred)**2 / num_of_bins)

        return loss

    return custom_loss
