import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import pickle
import random
from sklearn.preprocessing import StandardScaler
import json
import h5py
import os
import torch.nn as nn
from genetic_mutations_eep import Genetic_Mutations

from propy import PyPro
from propy import PseudoAAC
from propy import AAComposition
from propy import Autocorrelation
from propy import CTD
from propy import QuasiSequenceOrder

from modlamp.descriptors import GlobalDescriptor

feldan_ft_v0 = pd.read_excel("Feldan (filled, non-standardized).xlsx", index_col = 0)
feldan_ft_nozeros = feldan_ft_v0.loc[:, (feldan_ft_v0 != 0).any(axis=0)] # Remove columns where all values are zero

feldan_cor_matrix = feldan_ft_nozeros.corr().abs()
feldan_upper_tri = feldan_cor_matrix.where(np.triu(np.ones(feldan_cor_matrix.shape), k=1).astype(np.bool))
feldan_to_drop = [column for column in feldan_upper_tri.columns if any(feldan_upper_tri[column] > 0.95)]
feldan_ft_noncor = feldan_ft_nozeros.drop(feldan_to_drop, axis=1)

scaler = StandardScaler()
feldan_ft_noncor[feldan_ft_noncor.columns] = scaler.fit_transform(feldan_ft_noncor[feldan_ft_noncor.columns])
feldan_ft = feldan_ft_noncor

class Predictor:
    def __init__(self, **kwargs):
        '''
        Initialize a Predictor object which can be used in both modes - train and predict
            
        Using a pre-trained model - 
        
            predictor.Predictor(
                model_path = '/path/to/model.hdf5',
                stats_path = '/path/to/folder/',
                seq_max = 108
            )
            
            predictor.predict('NEWSEQ')

        Parameters
        ----------
                    
        model_path:     str
                        filepath of pre-trained model, to be used when using for prediction
        
        stats_path:     str
                        filepath of training dataset statistics

        seq_max:        int
                        maximum permissible length of sequence in predictor
                    
        '''

        self.__model_path = kwargs.get('model_path')
        self.__stats_path = kwargs.get('stats_path')
        self.__seq_max = kwargs.get('seq_max')

        with open(self.__model_path, 'rb') as f:
            self.model = pickle.load(f)

    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''
    def cf_helicity(self, my_sequence):
      # Define the Chou-Fasman parameters
      alpha_helix_params = {
          'A': [142,  83,  66, 106, 130,  84, 116,  54,  78,  60],
          'C': [ 99, 119, 119, 167, 153,  96, 121,  56, 119,  18],
          'D': [101,  47,  57,  75, 163,  84,  97, 163,  37,  59],
          'E': [151,  25,  37,  56, 194,  84,  74, 194,  37,  49],
          'F': [113, 138, 138, 187, 135, 114, 137,  58, 113,  13],
          'G': [ 57,  75,  75,  75,  75,  57,  75,  37,  60,  60],
          'H': [ 83,  66,  83,  96, 167,  73, 105,  26,  83,  20],
          'I': [108, 160, 160, 151, 131,  97, 131,  43, 108,  15],
          'K': [121,  54,  57,  87, 159,  60, 117, 159,  31,  56],
          'L': [121, 130, 130, 151, 131,  97, 131,  43, 108,  13],
          'M': [137, 105, 105, 137, 170, 114, 137,  15, 137,   8],
          'N': [ 81,  47,  47,  60, 163,  60,  75, 163,  31,  47],
          'P': [ 57,  75,  75,  75,  75,  57,  75,  37,  60,  60],
          'Q': [111,  36,  47,  56, 180,  84,  87, 180,  31,  42],
          'R': [114,  54,  57,  75, 141,  60, 105, 141,  31,  56],
          'S': [ 77,  60,  54,  54, 120,  54,  60, 120,  26,  42],
          'T': [ 83,  97,  83,  83, 102,  97,  83,  56,  83,  31],
          'V': [ 99, 170, 170, 142, 105,  73, 105,  26,  99,  13],
          'W': [108, 137, 137, 186, 108, 114, 167,  35, 108,  10],
          'Y': [ 83, 138, 138, 173,  99,  84, 147,  47,  83,  12]
      }

      #Define the input sequence
      sequence = my_sequence

      #Initialize the helix probability parameter for each residue to 0
      helix_probs = np.zeros(len(sequence))

      #Iterate over the residues in the sequence
      for i, residue in enumerate(sequence):
        # Look up the Chou-Fasman parameters for the residue
        params = alpha_helix_params.get(residue)
        if params is None:
          # If the residue is not in the Chou-Fasman table, set the helix probability to 0
          helix_probs[i] = 0
        else:
          # Calculate the helix probability parameter for the residue
          helix_param = sum(params)
        helix_probs[i] = helix_param

      helical_content = 0.001*sum(helix_probs)/len(sequence)

      return helical_content

    def featurize(self, sequence, task):
        '''
        Public function to get feature vector for peptide sequence
        
        Parameters
        -------
        sequence:   str
                    peptide/polymer sequence

        task:           str (Values: '% RFP+', '% GFP+', 'EV Score')
                        Label to be optimized
        Returns
        -------
        features

        '''
        sequence = sequence.replace(" ", "")
        ft_dataframe = pd.DataFrame()
        
        while True:
          try:
            peptide_ft = PyPro.GetProDes(sequence).GetALL(paac_lamda = 10, paac_weight = 0.5, apaac_lamda = 10, apaac_weight = 0.5, socn_maxlag = 10, qso_maxlag = 10)
            break
          except:
            sequence = Genetic_Mutations.mutate(sequence)
            peptide_ft = PyPro.GetProDes(sequence).GetALL(paac_lamda = 10, paac_weight = 0.5, apaac_lamda = 10, apaac_weight = 0.5, socn_maxlag = 10, qso_maxlag = 10)

        for entry in peptide_ft:
            ft_dataframe.loc["Peptide", entry] = peptide_ft[entry]

        desc = GlobalDescriptor(sequence)
        desc.calculate_all(amide = False)
        ft_dataframe.loc["Peptide", "Sequence Length"] = desc.descriptor[0][0]
        ft_dataframe.loc["Peptide", "Sequence Charge"] = desc.descriptor[0][2]
        ft_dataframe.loc["Peptide", "Charge Density"] = desc.descriptor[0][3]
        ft_dataframe.loc["Peptide", "Isoelectric Point"] = desc.descriptor[0][4]
        ft_dataframe.loc["Peptide", "Instability Index"] = desc.descriptor[0][5]
        ft_dataframe.loc["Peptide", "Aromaticity"] = desc.descriptor[0][6]
        ft_dataframe.loc["Peptide", "Aliphatic Index"] = desc.descriptor[0][7]
        ft_dataframe.loc["Peptide", "Boman Index"] = desc.descriptor[0][8]
        ft_dataframe.loc["Peptide", "Instability Index"] = desc.descriptor[0][9] # MISTAKE
        ft_dataframe.loc["Peptide", "Helical Content"] = self.cf_helicity(sequence)#-0.8)/0.2
        
        nonzeros = np.where((feldan_ft_v0 != 0).any(axis=0))
        ft_dataframe_nozeros = ft_dataframe.iloc[:, nonzeros[0]]

        ft_dataframe_noncor = ft_dataframe_nozeros.drop(feldan_to_drop, axis=1)

        ft_dataframe_noncor[ft_dataframe_noncor.columns] = scaler.transform(ft_dataframe_noncor[ft_dataframe_noncor.columns])
        ft_dataframe = ft_dataframe_noncor

        if task == '% RFP+ lasso':
            with open('MLP Regressor (no viability, no self-training).sav', "rb") as f:
                mlp = pickle.load(f)

            mlp.output_layer = nn.Identity()
            # Freeze the layers in the pre-trained model
            for param in mlp.parameters():
                param.requires_grad = False
            mlp.eval()

            features = mlp(ft_dataframe).detach().numpy()

        elif task == '% RFP+ mlp':
            with open('MLP Regressor (no viability, no self-training).sav', "rb") as f:
                mlp = pickle.load(f)

            mlp.output_layer = nn.Identity()
            # Freeze the layers in the pre-trained model
            for param in mlp.parameters():
                param.requires_grad = False
            mlp.eval()

            features = mlp(ft_dataframe)


        elif task == "% GFP+" or task == "EV Score":
            features = ft_dataframe

        else:
            print(f"{task} is not a valid task")

        return features

    def make_prediction(self, sequence, task):
        '''
        Public function to predict the activity
        
        Parameters
        -------
        sequence:   str
                    peptide/polymer sequence
        
        task:           str (Values: '% RFP+', '% GFP+', 'EV Score')
                        Label to be optimized

        Returns
        -------
        y:  float
            predicted efficacy

        '''
        
        sequence_ft = self.featurize(sequence, task)

        if task == '% RFP+ lasso':
            prediction = self.model.predict(sequence_ft)

        else:
            prediction = self.model(sequence_ft).detach().numpy()[0][0]

        return prediction
