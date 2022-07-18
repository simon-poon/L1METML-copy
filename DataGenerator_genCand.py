
import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot
import awkward as ak
from utils import convertXY2PtPhi, preProcessing, to_np_array
import h5py
import os
import itertools


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_files, batch_size=1024, n_dim=100, maxNPF=100, compute_ef=0,
                 max_entry=100000000, edge_list=[]):
        'Initialization'
        self.n_features_pf = 6
        self.n_features_pf_cat = 2
        self.normFac = 1.
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.n_channels = 8
        self.global_IDs = []
        self.local_IDs = []
        self.file_mapping = []
        self.max_entry = max_entry
        self.open_files = [None]*len(list_files)
        self.maxNPF = maxNPF
        self.compute_ef = compute_ef
        self.edge_list = edge_list
        running_total = 0

        self.h5files = []
        for ifile in list_files:
            print(ifile)
            varList = [
                    'nGenCands', 'GenCands_pt', 'GenCands_eta', 'GenCands_phi',
                    'GenCands_mass', 'GenCands_pdgId'
                    ]
            #with uproot.open(ifile) as upfile:

            '''arr = upfile['Events;1']['genMet_pt'].array(library='np')
            arr2 = upfile['Events;1']['genMet_phi'].array(library='np')
            arr3 = upfile['Events;1']['nGenCands'].array(library='np')
            print('genMet_pt:  ', np.shape(arr))
            print('genMet_phi:  ', np.shape(arr2))
            print('nGenCands:  ', np.shape(arr3))'''

            h5file_path = ifile.replace('.root', '.h5')

            
            varList = [
                    'nGenCands', 'GenCands_pt', 'GenCands_eta', 'GenCands_phi',
                    'GenCands_mass', 'GenCands_charge', 'GenCands_pdgId'
                    ]

            if not os.path.isfile(h5file_path):
                os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
            self.h5files.append(h5file_path)

            with uproot.open(ifile) as upfile:
                tree = upfile['Events'].arrays(varList, entry_stop=-1)
                numEvents = len(tree['nGenCands'])
                maxCands = 1500
                nFeatures = 6

                # input Puppi candidates
                # recoil estimators

                pt = ak.pad_none( tree['GenCands_pt'], maxCands)    # makes sure each GenCands array is the same size
                pt_np = ak.to_numpy(pt)    # convert from awkward array to numpy
                eta = ak.pad_none( tree['GenCands_eta'], maxCands)
                eta_np = ak.to_numpy(eta)
                phi = ak.pad_none( tree['GenCands_phi'], maxCands)
                phi_np = ak.to_numpy(phi)
                mass = ak.pad_none( tree['GenCands_mass'], maxCands)
                mass_np = ak.to_numpy(mass)
                charge = ak.pad_none( tree['GenCands_charge'], maxCands)
                charge_np = ak.to_numpy(charge)
                pdgId = ak.pad_none( tree['GenCands_pdgId'], maxCands)
                pdgId_np = ak.to_numpy(pdgId)


                Y_GenCand = np.zeros(shape=(numEvents, maxCands, 6), dtype=float, order='F')
                Y_GenCand[:, :, 0] += pt_np
                Y_GenCand[:, :, 1] += eta_np
                Y_GenCand[:, :, 2] += phi_np
                Y_GenCand[:, :, 3] += mass_np
                Y_GenCand[:, :, 4] += charge_np
                Y_GenCand[:, :, 5] += pdgId_np

            with h5py.File(h5file_path, 'a') as h5f:
                h5f.require_dataset('Y_GenCand', shape=np.shape(Y_GenCand), dtype=Y_GenCand.dtype, data=Y_GenCand, exact=False, compression='lzf')                     # assign new values to data
                print('hi')
                print("keys:  ", h5f.keys())
                print("X shape:  ", np.shape(h5f['X'][0:-1]))
                print("Y shape:  ", np.shape(h5f['Y'][0:-1]))
                print("Y_GenCand shape:  ", np.shape(h5f['Y_GenCand'][0:-1]))
                    

        for i, file_name in enumerate(self.h5files):
            with h5py.File(file_name, "r") as h5_file:
                self.open_files.append(h5_file)
                nEntries = len(h5_file['X'])
                self.global_IDs.append(np.arange(running_total, running_total+nEntries))
                self.local_IDs.append(np.arange(0, nEntries))
                self.file_mapping.append(np.repeat([i], nEntries))
                running_total += nEntries
                h5_file.close()
        self.global_IDs = np.concatenate(self.global_IDs)
        self.local_IDs = np.concatenate(self.local_IDs)
        self.file_mapping = np.concatenate(self.file_mapping)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.global_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        files = self.file_mapping[index*self.batch_size:(index+1)*self.batch_size]
        unique_files = np.unique(files)
        starts = np.array([min(indexes[files == i]) for i in unique_files])
        stops = np.array([max(indexes[files == i]) for i in unique_files])

        # Check if files needed open (if not open them)
        # Also if file is not needed, close it
        for ifile, file_name in enumerate(self.h5files):
            if ifile in unique_files:
                if self.open_files[ifile] is None:
                    self.open_files[ifile] = h5py.File(file_name, "r")
            else:
                if self.open_files[ifile] is not None:
                    self.open_files[ifile].close()
                    self.open_files[ifile] = None

        # Generate data
        return self.__data_generation(unique_files, starts, stops)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.local_IDs

    def deltaR_calc(self, eta1, phi1, eta2, phi2):
        """ calculate deltaR """
        dphi = (phi1-phi2)
        gt_pi_idx = (dphi > np.pi)
        lt_pi_idx = (dphi < -np.pi)
        dphi[gt_pi_idx] -= 2*np.pi
        dphi[lt_pi_idx] += 2*np.pi
        deta = eta1-eta2
        return np.hypot(deta, dphi)
    def kT_calc(self, pti, ptj, dR):
        min_pt = np.minimum(pti, ptj)
        kT = min_pt * dR
        return kT
    def z_calc(self, pti, ptj):
        epsilon = 1.0e-12
        min_pt = np.minimum(pti, ptj)
        z = min_pt/(pti + ptj + epsilon)
        return z
    def mass2_calc(self, pi, pj):
        pij = pi + pj
        m2 = pij[:,:,0]**2 - pij[:, :, 1]**2 - pij[:, :, 2]**2 - pij[:, :, 3]**2
        return m2

    def __data_generation(self, unique_files, starts, stops):
        'Generates data containing batch_size samples'
        # X : (n_samples, n_dim, n_channels)
        # y : (n_samples, 2)
        Xs = []
        ys = []

        # Generate data
        for ifile, start, stop in zip(unique_files, starts, stops):
            self.X, self.y = self.__get_features_labels(ifile, start, stop)

            '''print(np.shape(self.X))
            count80 = 0
            count85 = 0
            count90 = 0
            for batch_idx in range(self.batch_size):
                if (np.all(self.X[batch_idx,80,:] == np.zeros(8))):
                    count80 += 1
                    count85 += 1
                    count90 += 1
                elif (np.all(self.X[batch_idx,85,:] == np.zeros(8))):
                    count85 += 1
                    count90 += 1
                elif (np.all(self.X[batch_idx,90,:] == np.zeros(8))):
                    count90 += 1
            print('count80:  ', count80)
            print('count85:  ', count85)
            print('count90:  ', count90)'''

            Xs.append(self.X)
            ys.append(self.y)

        # Stack data if going over multiple files
        if len(unique_files) > 1:
            self.X = np.concatenate(Xs, axis=0)
            self.y = np.concatenate(ys, axis=0)

        # process inputs
        Y = self.y / (-self.normFac)
        Xi, Xp, Xc1, Xc2 = preProcessing(self.X, self.normFac)

        N = self.maxNPF
        Nr = N*(N-1)

        if self.compute_ef == 1:
            eta = Xi[:, :, 1]
            phi = Xi[:, :, 2]
            pt = Xi[:, :, 0]
            if ('m2' in self.edge_list):
                px = Xp[:, :, 0]
                py = Xp[:, :, 1]
                pz = pt*np.sinh(eta)
                energy = np.sqrt(px**2 + py**2 + pz**2)
                p4 = np.stack((energy, px, py, pz), axis=-1)
            receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
            edge_idx = np.array(receiver_sender_list)
            edge_stack = []
            if ('dR' in self.edge_list) or ('kT' in self.edge_list):
                eta1 = eta[:, edge_idx[:,0]]
                phi1 = phi[:, edge_idx[:,0]]
                eta2 = eta[:, edge_idx[:,1]]
                phi2 = phi[:, edge_idx[:,1]]
                dR = self.deltaR_calc(eta1, phi1, eta2, phi2)
                edge_stack.append(dR)
            if ('kT' in self.edge_list) or ('z' in self.edge_list):
                pt1 = pt[:, edge_idx[:,0]]
                pt2 = pt[:, edge_idx[:,1]]
                if ('kT' in self.edge_list):
                    kT = self.kT_calc(pt1, pt2, dR)
                    edge_stack.append(kT)
                if ('z' in self.edge_list):
                    z = self.z_calc(pt1, pt2)
                    edge_stack.append(z)
            if ('m2' in self.edge_list):
                p1 = p4[:, edge_idx[:,0], :]
                p2 = p4[:, edge_idx[:,1], :]
                m2 = self.mass2_calc(p1, p2)
                edge_stack.append(m2)
            ef = np.stack(edge_stack, axis=-1)
            
            
            '''
            eta = Xi[:, :, 1:2]
            phi = Xi[:, :, 2:3]
            pt = Xi[:, :, 0:1]
            if ('m2' in self.edge_list):
                px = Xp[:, :, 0:1]
                py = Xp[:, :, 1:2]
                pz = pt*np.sinh(eta)
                energy = np.sqrt(px**2 + py**2 + pz**2)
                p4 = np.concatenate((energy, px, py, pz), axis=-1)
            receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
            set_size = Xi.shape[0]
            ef = np.zeros([set_size, Nr, len(self.edge_list)])     # edge features: dimensions of [# of events, # of edges, # of edges]
            for count, edge in enumerate(receiver_sender_list):       # for loop creates edge features
                receiver = edge[0]  # "receiver_sender_list" generates edge and receiving indices
                sender = edge[1]
                if ('dR' in self.edge_list) or ('kT' in self.edge_list):
                    eta1 = eta[:, receiver, :]
                    phi1 = phi[:, receiver, :]
                    eta2 = eta[:, sender, :]
                    phi2 = phi[:, sender, :]
                    dR = self.deltaR_calc(eta1, phi1, eta2, phi2)
                    ef[:, count, 0:1] = dR
                if ('kT' in self.edge_list) or ('z' in self.edge_list):
                    pt1 = pt[:, receiver, :]
                    pt2 = pt[:, sender, :]
                    if ('kT' in self.edge_list):
                        kT = self.kT_calc(pt1, pt2, dR)
                        ef[:, count, 1:2] = kT
                    if ('z' in self.edge_list):
                        z = self.z_calc(pt1, pt2)
                        ef[:, count, 2:3] = z
                if ('m2' in self.edge_list):
                    p1 = p4[:, receiver, :]
                    p2 = p4[:, sender, :]
                    m2 = self.mass2_calc(p1, p2)
                    ef[:, count, 3:4] = m2'''

            '''print('dR shape')
                print(dR.shape)
                print('-----')
                print('kT shape')
                print(kT.shape)
                print('-----')
                print('ef shape')
                print(ef.shape)
                print('-----')
                print('Xi shape')
                print(Xi.shape)
                ef[:,count,2:3] = z'''

            Xc = [Xc1, Xc2]
            # dimension parameter for keras model
            self.emb_input_dim = {i: int(np.max(Xc[i][0:1000])) + 1 for i in range(self.n_features_pf_cat)}
            #print('emb_input_dim:  ', self.emb_input_dim)

            # Prepare training/val data
            Yr = Y
            Xr = [Xi, Xp] + Xc + [ef]
            return Xr, Yr

        else:
            Xc = [Xc1, Xc2]
            # dimension parameter for keras model
            self.emb_input_dim = {i: int(np.max(Xc[i][0:1000])) + 1 for i in range(self.n_features_pf_cat)}

            # Prepare training/val data
            Yr = Y
            Xr = [Xi, Xp] + Xc
            return Xr, Yr

    def __get_features_labels(self, ifile, entry_start, entry_stop):
        'Loads data from one file'

        # Double check that file is open
        if self.open_files[ifile] is None:
            h5_file = h5py.File(file_name, "r")
        else:
            h5_file = self.open_files[ifile]

        X = h5_file['X'][entry_start:entry_stop+1]
        y = h5_file['Y'][entry_start:entry_stop+1]

        if self.maxNPF < 100:            
            order = X[:,:,0].argsort(axis=1)[:,::-1]
            shape = np.shape(X)
            for x in range(shape[0]):
                X[x,:,:] = X[x,order[x],:]
            X = X[:, 0:self.maxNPF, :]

        return X, y
    
    '''        
    def deltaR_calc(self, eta1, phi1, eta2, phi2):
        """ calculate deltaR """
        dphi = (phi1-phi2)
        gt_pi_idx = (dphi > np.pi)
        lt_pi_idx = (dphi < -np.pi)
        dphi[gt_pi_idx] -= 2*np.pi
        dphi[lt_pi_idx] += 2*np.pi
        deta = eta1-eta2
        return np.hypot(deta, dphi)

    def kT_calc(self, pti, ptj, dR):
        min_pt = np.minimum(pti[:, 0:1], ptj[:, 0:1])
        kT = min_pt * dR
        return kT

    def z_calc(self, pti, ptj):
        epsilon = 1.0e-12
        min_pt = np.minimum(pti[:, 0:1], ptj[:, 0:1])
        z = min_pt/(pti + ptj + epsilon)
        return z

    def mass2_calc(self, pi, pj):
        pij = pi + pj
        m2 = pij[:, 0:1]**2 - pij[:, 1:2]**2 - pij[:, 2:3]**2 - pij[:, 3:4]**2
        return m2
    '''

