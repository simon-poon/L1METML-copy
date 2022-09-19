import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
import math
#import setGPU
import time
import os
import pathlib
import datetime
import tqdm
import h5py
from glob import glob
import itertools

# Import custom modules

from Write_MET_binned_histogram import *
from cyclical_learning_rate import CyclicLR
from models import *
from utils import *
from loss import custom_loss
from DataGenerator import DataGenerator

import matplotlib.pyplot as plt
import mplhep as hep


def MakeEdgeHist(edge_feat, xname, outputname, nbins=1000, density=False, yname="# of edges"):
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10, 8))
    plt.hist(edge_feat, bins=nbins, density=density, histtype='step', facecolor='k', label='Truth')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()


'''def MakeEdgeHist_nozeros(edge_feat, xname, outputname, nbins=1000, density=False, yname="# of edges"):
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(20, 16))
    plt.hist(edge_feat, bins=nbins, range=(1e-12,100), density=density, histtype='step', facecolor='k', label='Truth')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()'''


def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    gt_pi_idx = (dphi > np.pi)
    lt_pi_idx = (dphi < -np.pi)
    dphi[gt_pi_idx] -= 2*np.pi
    dphi[lt_pi_idx] += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


def get_callbacks(path_out, sample_size, batch_size):
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=False)

    csv_logger = CSVLogger(f'{path_out}loss_history.log')

    # model checkpoint callback
    # this saves our model architecture + parameters into model.h5
    model_checkpoint = ModelCheckpoint(f'{path_out}model.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto',
                                       period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

    lr_scale = 1.
    clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=sample_size/batch_size, mode='triangular2')

    stop_on_nan = tensorflow.keras.callbacks.TerminateOnNaN()

    callbacks = [early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint]

    return callbacks


def test(Yr_test, predict_test, PUPPI_pt, path_out):

    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out=path_out)

    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    extract_result(predict_test, Yr_test, path_out, 'TTbar', 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, 'TTbar', 'PU')

    MET_rel_error_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], name=''+path_out+'rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], 20, 0, 500, 0, '.', name=''+path_out+'PrVSGen.png')

    Phi_abs_error_opaque(PUPPI_pt[:, 1], predict_test[:, 1], Yr_test[:, 1], name=path_out+'Phi_abs_err')
    Pt_abs_error_opaque(PUPPI_pt[:, 0], predict_test[:, 0], Yr_test[:, 0], name=path_out+'Pt_abs_error')


def train_dataGenerator(args):
    # general setup
    maxNPF = args.maxNPF
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = args.epochs
    batch_size = args.batch_size
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized
    model = args.model
    units = list(map(int, args.units))
    compute_ef = args.compute_edge_feat
    edge_list = args.edge_features

    # separate files into training, validation, and testing
    filesList = glob(os.path.join(inputPath, '*.root'))
    filesList.sort(reverse=True)

    assert len(filesList) >= 3, "Need at least 3 files for DataGenerator: 1 valid, 1 test, 1 train"

    valid_nfiles = max(1, int(.1*len(filesList)))
    train_nfiles = len(filesList) - 2*valid_nfiles
    test_nfiles = valid_nfiles
    train_filesList = filesList[0:train_nfiles]
    valid_filesList = filesList[train_nfiles: train_nfiles+valid_nfiles]
    test_filesList = filesList[train_nfiles+valid_nfiles:test_nfiles+train_nfiles+valid_nfiles]

    if compute_ef == 1:

        # set up data generators; they perform h5 conversion if necessary and load in data batch by batch
        trainGenerator = DataGenerator(list_files=train_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        validGenerator = DataGenerator(list_files=valid_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        testGenerator = DataGenerator(list_files=test_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        Xr_train, Yr_train = trainGenerator[0]  # this apparenly calls all the attributes, so that we can get the correct input dimensions (train_generator.emb_input_dim)
        print('len_of_trainGenerator', len(trainGenerator))

        ''''
        dR_none_conc = np.array([0])
        kT_none_conc = np.array([0])
        z_none_conc = np.array([0])
        m2_none_conc = np.array([0])

        dR_none_val_conc = np.array([0])
        kT_none_val_conc = np.array([0])
        z_none_val_conc = np.array([0])
        m2_none_val_conc = np.array([0])

        for fourth_ind in range(30):

            # The next 50 or so lines (up until the else) generates histograms of inputs
            #first_index = np.random.randint(0,high=2637)
            #fourth_index = np.random.randint(0,high=epochs)

            dR = trainGenerator[0][0][4][fourth_ind, :, 0]
            kT = trainGenerator[0][0][4][fourth_ind, :, 1]
            z = trainGenerator[0][0][4][fourth_ind, :, 2]
            #m2 = trainGenerator[0][0][4][fourth_ind, :, 3]

            dR_val = validGenerator[0][0][4][fourth_ind, :, 0]
            kT_val = validGenerator[0][0][4][fourth_ind, :, 1]
            z_val = validGenerator[0][0][4][fourth_ind, :, 2]
            #m2_val = validGenerator[0][0][4][fourth_ind, :, 3]

            # No zero-padded w/ zero-padded interaction
            withzeros = trainGenerator[0][0][4][fourth_ind, :, :]
            nozeros = withzeros[~np.all(withzeros == 0., axis=1)]
            dR_nozeros = nozeros[:, 0]
            kT_nozeros = nozeros[:, 1]
            z_nozeros = nozeros[:, 2]
            #m2_nozeros = nozeros[:, 3]

            withzeros_val = validGenerator[0][0][4][fourth_ind, :, :]
            nozeros_val = withzeros_val[~np.all(withzeros_val == 0., axis=1)]
            dR_nozeros_val = nozeros_val[:, 0]
            kT_nozeros_val = nozeros_val[:, 1]
            z_nozeros_val = nozeros_val[:, 2]
            #m2_nozeros_val = nozeros_val[:, 3]

            # No particle w/ zero-padded interaction
            data_bool = np.array(nozeros, dtype=bool)
            b = [True, False, False]
            delete = nozeros[~np.all(data_bool == b, axis=1)]
            dR_none = delete[:, 0]
            kT_none = delete[:, 1]
            z_none = delete[:, 2]
            #m2_none = delete[:, 3]

            data_bool_val = np.array(nozeros_val, dtype=bool)
            delete_val = nozeros_val[~np.all(data_bool_val == b, axis=1)]
            dR_none_val = delete_val[:, 0]
            kT_none_val = delete_val[:, 1]
            z_none_val = delete_val[:, 2]
            #m2_none_val = delete_val[:, 3]

            dR_none_conc = np.concatenate((dR_none_conc, dR_none))
            kT_none_conc = np.concatenate((kT_none_conc, kT_none))
            z_none_conc = np.concatenate((z_none_conc, z_none))
            #m2_none_conc = np.concatenate((m2_none_conc, m2_none))

            dR_none_val_conc = np.concatenate((dR_none_val_conc, dR_none_val))
            kT_none_val_conc = np.concatenate((kT_none_val_conc, kT_none_val))
            z_none_val_conc = np.concatenate((z_none_val_conc, z_none_val))
            #m2_none_val_conc = np.concatenate((m2_none_val_conc, m2_none_val))

            for index1 in first_index[1:]:
                for index4 in fourth_index[1:]:
                    new_dR = trainGenerator[index1][0][4][index4,:,0]
                    new_kT = trainGenerator[index1][0][4][index4,:,1]
                    new_z = trainGenerator[index1][0][4][index4,:,2]
                    dR = np.concatenate((dR, new_dR), axis=0)
                    kT = np.concatenate((kT, new_kT), axis=0)
                    z = np.concatenate((z, new_z), axis=0)

        MakeEdgeHist(dR, xname='dR', outputname=f'{path_out}dR.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(dR_nozeros, xname='dR', outputname=f'{path_out}dR_nozeros.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(dR_none_conc, xname='dR', outputname=f'{path_out}dR_none.png', nbins=100, density=False, yname="# of edges")

        MakeEdgeHist(kT, xname='kT', outputname=f'{path_out}kT.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(kT_nozeros, xname='kT', outputname=f'{path_out}kT_nozeros.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(kT_none_conc, xname='kT', outputname=f'{path_out}kT_none.png', nbins=100, density=False, yname="# of edges")

        MakeEdgeHist(z, xname='z', outputname=f'{path_out}z.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(z_nozeros, xname='z', outputname=f'{path_out}z_nozeros.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(z_none_conc, xname='z', outputname=f'{path_out}z_none.png', nbins=100, density=False, yname="# of edges")

        #MakeEdgeHist(m2, xname='m2', outputname=f'{path_out}m2.png', nbins=100, density=False, yname="# of edges")
        #MakeEdgeHist(m2_nozeros, xname='m2', outputname=f'{path_out}m2_nozeros.png', nbins=100, density=False, yname="# of edges")
        #MakeEdgeHist(m2_none_conc, xname='m2', outputname=f'{path_out}m2_none.png', nbins=100, density=False, yname="# of edges")

        MakeEdgeHist(dR_none_val_conc, xname='dR', outputname=f'{path_out}dR_none_val.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(kT_none_val_conc, xname='kT', outputname=f'{path_out}kT_none_val.png', nbins=100, density=False, yname="# of edges")
        MakeEdgeHist(z_none_val_conc, xname='z', outputname=f'{path_out}z_none_val.png', nbins=100, density=False, yname="# of edges")
        #MakeEdgeHist(m2_none_val_conc, xname='m2', outputname=f'{path_out}m2_none_val.png', nbins=100, density=False, yname="# of edges")
        '''

    else:
        trainGenerator = DataGenerator(list_files=train_filesList, batch_size=batch_size)
        validGenerator = DataGenerator(list_files=valid_filesList, batch_size=batch_size)
        testGenerator = DataGenerator(list_files=test_filesList, batch_size=batch_size)
        Xr_train, Yr_train = trainGenerator[0]  # this apparenly calls all the attributes, so that we can get the correct input dimensions (train_generator.emb_input_dim)

    # Load training model
    if quantized is None:
        if model == 'dense_embedding':
            keras_model = dense_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          t_mode=t_mode,
                                          with_bias=False,
                                          units=units)
        elif model == 'graph_embedding':
            teacher = graph_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          units=units, compute_ef=compute_ef, edge_list=edge_list)
            student = dense_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          t_mode=t_mode,
                                          with_bias=False,
                                          units=units)

        elif model == 'node_select':
            keras_model = node_select(n_features=n_features_pf,
                                      emb_out_dim=2,
                                      n_features_cat=n_features_pf_cat,
                                      activation='tanh',
                                      embedding_input_dim=trainGenerator.emb_input_dim,
                                      number_of_pupcandis=maxNPF,
                                      units=units, compute_ef=compute_ef)
                                      

    else:
        logit_total_bits = int(quantized[0])
        logit_int_bits = int(quantized[1])
        activation_total_bits = int(quantized[0])
        activation_int_bits = int(quantized[1])

        keras_model = dense_embedding_quantized(n_features=n_features_pf,
                                                emb_out_dim=2,
                                                n_features_cat=n_features_pf_cat,
                                                activation_quantizer='quantized_relu',
                                                embedding_input_dim=trainGenerator.emb_input_dim,
                                                number_of_pupcandis=maxNPF,
                                                t_mode=t_mode,
                                                with_bias=False,
                                                logit_quantizer='quantized_bits',
                                                logit_total_bits=logit_total_bits,
                                                logit_int_bits=logit_int_bits,
                                                activation_total_bits=activation_total_bits,
                                                activation_int_bits=activation_int_bits,
                                                alpha=1,
                                                use_stochastic_rounding=False,
                                                units=units)

    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    print("---- Distillation ----")
    if t_mode == 0:
        teacher.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        teacher.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    # Run training
    print(teacher.summary())


    start_time = time.time()  # check start time
    history = teacher.fit(trainGenerator,
                              epochs=epochs,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=validGenerator,
                              callbacks=get_callbacks(path_out, len(trainGenerator), batch_size))
    end_time = time.time()  # check end time

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(optimizer=optimizer,
                    metrics=keras.metrics.MeanSquaredError(),
                    student_loss_fn=keras.losses.MeanSquaredError(),
                    distillation_loss_fn=keras.losses.MeanSquaredError(),
                    alpha=0.1,
                    temperature=10)
    distiller.fit(trainGenerator, epochs=2)

    predict_test = distiller.predict(testGenerator) * normFac
    all_PUPPI_pt = []
    Yr_test = []

    all_PUPPI_pt = []
    Yr_test = []
    for (Xr, Yr) in tqdm.tqdm(testGenerator):
        puppi_pt = np.sum(Xr[1], axis=1)
        all_PUPPI_pt.append(puppi_pt)
        Yr_test.append(Yr)

    PUPPI_pt = normFac * np.concatenate(all_PUPPI_pt)
    Yr_test = normFac * np.concatenate(Yr_test)

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()


def train_loadAllData(args):
    # general setup
    maxNPF = maxNPF
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = args.epochs
    batch_size = args.batch_size
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized
    units = list(map(int, args.units))
    compute_ef = args.compute_edge_feat
    model = args.model

    print('starting')

    # Read inputs
    # convert root files to h5 and store in same location
    h5files = []
    for ifile in glob(os.path.join(f'{inputPath}', '*.root')):
        h5file_path = ifile.replace('.root', '.h5')
        if not os.path.isfile(h5file_path):
            os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
        h5files.append(h5file_path)

    # It may be desireable to set specific files as the train, test, valid data sets
    # For now I keep train.py used: selection from a list of indicies

    print('right before Xorg')

    Xorg, Y = read_input(h5files)
    Y = Y / -normFac

    N = maxNPF
    Nr = N*(N-1)

    receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]

    if compute_ef == 1:
        print("Computing edge features")
        set_size = Xorg.shape[0]
        ef = np.zeros([set_size, Nr, 1])
        for count, edge in enumerate(receiver_sender_list):
            eta = Xorg[:, :, 3:4]
            phi = Xorg[:, :, 4:5]
            receiver = edge[0]
            sender = edge[1]
            eta1 = eta[:, receiver, :]
            phi1 = phi[:, receiver, :]
            eta2 = eta[:, sender, :]
            phi2 = phi[:, sender, :]
            dR = deltaR(eta1, phi1, eta2, phi2)
            ef[:, count, :] = dR
        print("edge features computed")

        Xi, Xp, Xc1, Xc2 = preProcessing(Xorg, normFac)
        Xc = [Xc1, Xc2]

        emb_input_dim = {
            i: int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
        }

        # Prepare training/val data
        Yr = Y
        Xr = [Xi, Xp] + Xc + [ef]

    else:
        print("else path")
        Xi, Xp, Xc1, Xc2 = preProcessing(Xorg, normFac)
        Xc = [Xc1, Xc2]

        emb_input_dim = {
            i: int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
        }

        # Prepare training/val data
        Yr = Y
        Xr = [Xi, Xp] + Xc

    indices = np.array([i for i in range(len(Yr))])
    indices_train, indices_test = train_test_split(indices, test_size=1./7., random_state=7)
    indices_train, indices_valid = train_test_split(indices_train, test_size=1./6., random_state=7)
    # roughly the same split as the data generator workflow (train:valid:test=5:1:1)

    Xr_train = [x[indices_train] for x in Xr]
    Xr_test = [x[indices_test] for x in Xr]
    Xr_valid = [x[indices_valid] for x in Xr]
    Yr_train = Yr[indices_train]
    Yr_test = Yr[indices_test]
    Yr_valid = Yr[indices_valid]

    # Load training model
    if quantized is None:
        if model == 'dense_embedding':
            keras_model = dense_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          t_mode=t_mode,
                                          with_bias=False,
                                          units=units)

        elif model == 'graph_embedding':
            keras_model = graph_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          units=units)

    else:
        logit_total_bits = int(quantized[0])
        logit_int_bits = int(quantized[1])
        activation_total_bits = int(quantized[0])
        activation_int_bits = int(quantized[1])

        keras_model = dense_embedding_quantized(n_features=n_features_pf,
                                                emb_out_dim=2,
                                                n_features_cat=n_features_pf_cat,
                                                activation_quantizer='quantized_relu',
                                                embedding_input_dim=emb_input_dim,
                                                number_of_pupcandis=maxNPF,
                                                t_mode=t_mode,
                                                with_bias=False,

                                                logit_quantizer='quantized_bits',
                                                logit_total_bits=logit_total_bits,
                                                logit_int_bits=logit_int_bits,
                                                activation_total_bits=activation_total_bits,
                                                activation_int_bits=activation_int_bits,
                                                alpha=1,
                                                use_stochastic_rounding=False,
                                                units=units)

    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    # Run training
    print(keras_model.summary())

    start_time = time.time()  # check start time
    history = keras_model.fit(Xr_train,
                              Yr_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=(Xr_valid, Yr_valid),
                              callbacks=get_callbacks(path_out, len(Yr_train), batch_size))

    end_time = time.time()  # check end time

    predict_test = keras_model.predict(Xr_test) * normFac
    PUPPI_pt = normFac * np.sum(Xr_test[1], axis=1)
    Yr_test = normFac * Yr_test

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()


def main():
    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workflowType',
        action='store',
        type=str,
        required=True,
        choices=[
            'dataGenerator',
            'loadAllData'],
        help='designate wheather youre using the data generator or loading all data into memory ')
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, required=True, help='designate output file path')
    parser.add_argument('--mode', action='store', type=int, required=True, choices=[0, 1], help='0 for L1MET, 1 for DeepMET')
    parser.add_argument('--epochs', action='store', type=int, required=False, default=100, help='number of epochs to train for')
    parser.add_argument('--batch-size', action='store', type=int, required=False, default=1024, help='batch size')
    parser.add_argument('--quantized', action='store', required=False, nargs='+', help='optional argument: flag for quantized model and specify [total bits] [int bits]; empty for normal model')
    parser.add_argument('--units', action='store', required=False, nargs='+', help='optional argument: specify number of units in each layer (also sets the number of layers)')
    parser.add_argument('--model', action='store', required=False, choices=['dense_embedding', 'graph_embedding', 'node_select'], default='dense_embedding', help='optional argument: model')
    parser.add_argument('--compute-edge-feat', action='store', type=int, required=False, choices=[0, 1], default=0, help='0 for no edge features, 1 to include edge features')
    parser.add_argument('--maxNPF', action='store', type=int, required=False, default=100, help='maximum number of PUPPI candidates')
    parser.add_argument('--edge-features', action='store', required=False, nargs='+', help='which edge features to use (i.e. dR, kT, z, m2)')

    args = parser.parse_args()
    workflowType = args.workflowType

    os.makedirs(args.output, exist_ok=True)

    if workflowType == 'dataGenerator':
        train_dataGenerator(args)
    elif workflowType == 'loadAllData':
        train_loadAllData(args)


if __name__ == "__main__":
    main()



