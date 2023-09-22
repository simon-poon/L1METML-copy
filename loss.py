import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def tfconvertXY2PtPhi(arrayXY):
    # convert from array with [:,0] as X and [:,1] as Y to [:,0] as pt and [:,1] as phi
    pt_mean = K.sqrt((arrayXY[:, 0]**2 + arrayXY[:, 1]**2) + 1e-10)
    phi_mean = tf.math.atan2(arrayXY[:, 1], 1e-10 + arrayXY[:, 0])
    arrayPtPhi = tf.stack([pt_mean,phi_mean],axis=-1)
    if arrayXY.get_shape()[-1] == 4:
        pt_25 = K.sqrt((arrayXY[:, 2]**2 + arrayXY[:, 3]**2) + 1e-10)
        phi_25 = tf.math.atan2(arrayXY[:, 3], 1e-10 + arrayXY[:, 2])
        #pt_75 = K.sqrt((arrayXY[:, 4]**2 + arrayXY[:, 5]**2) + 1e-10)
        #phi_75 = tf.math.atan2(arrayXY[:, 5], 1e-10 + arrayXY[:, 4])
        arrayPtPhi_quartile = tf.stack([pt_25, phi_25], axis=-1)
        arrayPtPhi = tf.concat([arrayPtPhi, arrayPtPhi_quartile], axis=-1)
    return arrayPtPhi


def custom_loss(y_true, y_pred):

    import tensorflow.keras.backend as K
    import tensorflow as tf

    px_truth = K.flatten(y_true[:, 0])
    py_truth = K.flatten(y_true[:, 1])
    px_pred = K.flatten(y_pred[:, 0])
    py_pred = K.flatten(y_pred[:, 1])
    #px_pred_25 = K.flatten(y_pred[:, 2])
    #py_pred_25 = K.flatten(y_pred[:, 3])
    #px_pred_75 = K.flatten(y_pred[:, 4])
    #py_pred_75 = K.flatten(y_pred[:, 5])

    ptphi_truth = tfconvertXY2PtPhi(y_true)
    pt_truth = ptphi_truth[:,0]
    phi_truth = ptphi_truth[:,1]
    ptphi_pred = tfconvertXY2PtPhi(y_pred)
    pt_pred_mean = ptphi_pred[:,0]
    phi_pred_mean = ptphi_pred[:,1]
    pt_pred_25 = ptphi_pred[:,2]
    phi_pred_25 = ptphi_pred[:,3]

    def resp_cor(pred_px, pred_py, truth_px, truth_py, pxpy=True):
        if pxpy==True:
            pt_truth = K.sqrt(truth_px*truth_px + truth_py*truth_py)

            #truth_px1 = truth_px / pt_truth
            #truth_py1 = truth_py / pt_truth

            # using absolute response
            # upar_pred = (truth_px1 * pred_px + truth_py1 * pred_py)/pt_truth
            upar_pred = K.sqrt(pred_px * pred_px + pred_py * pred_py) - pt_truth
        else:
            pt_truth = truth_px
            upar_pred = pred_px - pt_truth
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
        return dev

    def huber_loss(px_true, pred_px, py_true, pred_py, delta=1.0):
        px_error = px_true - pred_px
        py_error = py_true - pred_py
        px_abs_error = tf.abs(px_error)
        py_abs_error = tf.abs(py_error)
        quadratic_region = 0.5 * tf.square(px_abs_error) + tf.square(py_abs_error)
        linear_region = delta * px_abs_error + delta * py_abs_error - 0.5 * tf.square(delta)
        loss = tf.where(px_abs_error + py_abs_error < delta, quadratic_region, linear_region)
        return tf.reduce_mean(loss)

    def quantile_loss(y_true,y_quant,tau):
        z = y_true - y_quant
        loss = tf.where(z > 0, tau * z, (tau - 1) * z)
        return tf.reduce_mean(loss)

    
    sig_pt = tf.expand_dims(pt_pred_25, axis=-1)
    sig_phi = tf.expand_dims(phi_pred_25, axis=-1)
    pt_truth_expand = tf.expand_dims(pt_truth, axis=-1)
    #try pt axis and then x-y axis
    truth = tf.stack([pt_truth, phi_truth], axis=-1)
    truth = tf.expand_dims(truth,axis=1)
    pred = tf.stack([pt_pred_mean, phi_pred_mean], axis=-1)
    pred = tf.expand_dims(pred, axis=1)
    zeros = tf.zeros(shape=tf.shape(sig_pt))
    cov_mat_row1 = tf.concat([sig_pt+1e-12, zeros],axis=-1)
    cov_mat_row2 = tf.concat([zeros, sig_phi**2 + 1e-12],axis=-1)
    cov_mat = tf.stack([cov_mat_row1, cov_mat_row2], axis=-1)
    cov_mat_inv = tf.linalg.inv(cov_mat)
    #yuh = tf.linalg.matmul(cov_mat_inv, (truth-pred))
    #tf.print(yuh.get_shape())
    L = 0.5 * tf.linalg.matmul((truth - pred), tf.linalg.matmul(cov_mat_inv, tf.transpose((truth-pred), perm=[0,2,1]))) + 0.5 *tf.math.log(tf.linalg.det(cov_mat) + 1e-10)
    #tf.print(L.get_shape())
    L = tf.math.reduce_mean(L)


    return L
