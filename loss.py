import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def tfconvertXY2PtPhi(arrayXY):
    # convert from array with [:,0] as X and [:,1] as Y to [:,0] as pt and [:,1] as phi
    pt_mean = K.sqrt((arrayXY[:, 0]**2 + arrayXY[:, 1]**2) + 1e-10)
    phi_mean = tf.math.atan2(arrayXY[:, 1], 1e-10 + arrayXY[:, 0])
    arrayPtPhi = tf.stack([pt_mean,phi_mean],axis=-1)
    if arrayXY.get_shape()[-1] == 6:
        pt_25 = K.sqrt((arrayXY[:, 2]**2 + arrayXY[:, 3]**2) + 1e-10)
        phi_25 = tf.math.atan2(arrayXY[:, 3], 1e-10 + arrayXY[:, 2])
        pt_75 = K.sqrt((arrayXY[:, 4]**2 + arrayXY[:, 5]**2) + 1e-10)
        phi_75 = tf.math.atan2(arrayXY[:, 5], 1e-10 + arrayXY[:, 4])
        arrayPtPhi_quartile = tf.stack([pt_25, phi_25, pt_75, phi_75], axis=-1)
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
    pt_pred_75 = ptphi_pred[:,4]
    phi_pred_75 = ptphi_pred[:,5]

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

    delta = 1.0  # Huber loss delta
    tau_25 = 0.25  # 25% quantile
    tau_75 = 0.75  # 75% quantile
 
    huber_loss_value = 0.5*K.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)
    #huber_loss_value = huber_loss(px_truth, px_pred, py_truth, py_pred, delta)
    pt_quantile_loss_25 = quantile_loss(pt_truth, pt_pred_25, tau_25)
    pt_quantile_loss_75 = quantile_loss(pt_truth, pt_pred_75, tau_75)
    #phi_quantile_loss_25 = quantile_loss(phi_truth, phi_pred_25, tau_25)
    #phi_quantile_loss_75 = quantile_loss(phi_truth, phi_pred_75, tau_75)
    complete_loss_value = huber_loss_value + 5000*pt_quantile_loss_25 + 5000*pt_quantile_loss_75 # + phi_quantile_loss_25 + phi_quantile_loss_75
    dev_mean = resp_cor(px_pred, py_pred, px_truth, py_truth)
    #dev_25 = resp_cor(pt_pred_25, phi_pred_25, pt_truth, phi_truth, pxpy=False)
    #dev_75 = resp_cor(pt_pred_75, phi_pred_75, pt_truth, phi_truth, pxpy=False)
    #complete_loss_value += 5000.*dev
    loss = complete_loss_value + 5000.*dev_mean
    return loss
