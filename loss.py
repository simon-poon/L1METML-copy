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

        # using absolute response
        # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
        upar_pred = K.sqrt((bin_center_tf + px_pred)**2 + (bin_center_tf + py_pred)**2) - pt_truth
        pt_cut = pt_truth > 0.
        upar_pred = tf.boolean_mask(upar_pred, pt_cut)
        pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)
        
        #filter_bin0 = pt_truth_filtered < 50.
        filter_bin0 = tf.logical_and(pt_truth_filtered > 50.,  pt_truth_filtered < 100.)
        filter_bin1 = tf.logical_and(pt_truth_filtered > 100., pt_truth_filtered < 200.)
        filter_bin2 = tf.logical_and(pt_truth_filtered > 200., pt_truth_filtered < 300.)
        filter_bin3 = tf.logical_and(pt_truth_filtered > 300., pt_truth_filtered < 400.)
        filter_bin4 = pt_truth_filtered > 400.
        
        upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
        upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
        upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
        upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
        upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
        upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
        upar_pred_pos_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred > 0.))
        upar_pred_neg_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred < 0.))
        upar_pred_pos_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred > 0.))
        upar_pred_neg_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred < 0.))
        #upar_pred_pos_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred > 0.))
        #upar_pred_neg_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred < 0.))
        norm = tf.reduce_sum(pt_truth_filtered)
        dev = tf.abs(tf.reduce_sum(upar_pred_pos_bin0) + tf.reduce_sum(upar_pred_neg_bin0))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin1) + tf.reduce_sum(upar_pred_neg_bin1))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin2) + tf.reduce_sum(upar_pred_neg_bin2))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin3) + tf.reduce_sum(upar_pred_neg_bin3))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin4) + tf.reduce_sum(upar_pred_neg_bin4))
        #dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin5) + tf.reduce_sum(upar_pred_neg_bin5))
        dev /= norm
        
        loss = K.mean((px_truth - bin_center_tf - px_pred)**2 / num_of_bins + (py_truth - bin_center_tf - py_pred)**2 / num_of_bins)
        loss += 5000.*dev
        return loss

    return custom_loss
