import os
app_dir = os.getcwd()

import sys

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=FutureWarning)
import genetic_mutations_eep
import eep_predictor
import pickle
from IPython.display import display

import numpy as np
import pandas as pd
import json
import random
import math

class Optimizer:
    def __init__(self, **kwargs):
        '''
        Initialize a Generator object which can be used in both modes - train and generate
        
        Examples:
        optimizer = optimizer.Optimizer(
            model_path = '/path/to/model.sav',
            data_path = '/path/to/dataset.csv',
            stats_path = '/path/to/stats.json',
            seq_max = 50
            task = '% RFP+'
        )
        
        optimizer.optimize()

        Parameters
        ----------
        
        model_path:     str
                        filepath of pre-trained model, to be used when using for prediction

        data_path:      str
                        filepath of current dataset, to be used when training
                        Custom dataset as a *.csv file may be used. The dataset should contain a list of sequences and activity
                        The column headers should be 'sequences' and 'intensity'
                        
        stats_path:     str
                        filepath of training dataset statistics
                    
        seq_max:        int
                        maximum permissible length of sequence in predictor
                        
        max_attempts:   int, default: 100 (1000 used in original model)
                        maximum number of mutations per seed
        
        T:              float, default: 1
                        parameter used for simulated annealing

        task:           str (Values: '% RFP+ lasso', '% RFP+ mlp', '% GFP+', 'EV Score')
                        Label to be optimized
         
        '''

        self.__model_path = kwargs.get('model_path')
        self.__data_path = kwargs.get('data_path')
        self.__stats_path = kwargs.get('stats_path')
        self.__seq_max = kwargs.get('seq_max')
        self.__max_attempts = kwargs.get('max_attempts')
        
        self.task = kwargs.get('task')
        self.seq_list = pd.read_csv(self.__data_path)['Sequence'].tolist()
        
        self.predictor = eep_predictor.Predictor(
            model_path = self.__model_path, 
            stats_path = self.__stats_path,
            seq_max = self.__seq_max
        )

        self.genetic_mutations = genetic_mutations_eep.Genetic_Mutations(
            data_path = self.__data_path, 
            seq_max = self.__seq_max
        )
        
        with open(self.__stats_path) as f:
            self.dict_data = json.load(f)
    
    def __fitnessfunc(self, sequence):
        
        '''
        Utility function to evaluate the fitness of a mutated sequence against an objective function 

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        Returns
        -------
        value:   float
                 fitness of mutated sequence against the objective function
                 
        '''
        keys = list(self.dict_data.keys())
        value = (self.predictor.make_prediction(sequence, self.task) - self.dict_data[keys[0]]) / self.dict_data[keys[1]]
        return value

    def __genetic_algorithm(self):
        
        '''
        Utility function to implement the directed evolution
        
        '''

        oldseq = self.seq_df.at[self.i, 'seed']
        
        for attempt in range(self.__max_attempts):
            oldvalue = self.__fitnessfunc(oldseq)
            newseq = self.genetic_mutations.mutate(oldseq)
            newvalue = self.__fitnessfunc(newseq)

            if attempt % 10 == 0:
                print(attempt, oldseq, oldvalue)
                print(attempt, newseq, newvalue)

            delta = newvalue - oldvalue

            if (newvalue * np.exp(-delta*self.__T)) > oldvalue:
                oldseq = newseq
                self.seq_df.at[self.i, 'new_dict'][newseq] = [newvalue]
            else:
                continue

        print("Final: ", newseq, newvalue)

    def __post_process(self, new_seq_dict):
        '''
        Utility function to post process the mutations

        Parameters
        ----------
        new_seq_dict:   dict
                        dictionary of new sequences along with intensity and other parameters
                    
        Returns
        -------
        ga_df:  dataframe
                Sequence, Efficacy
                
        '''
            
        return new_seq_dict
    
        
    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''

    def optimize(self, list_seeds, **kwargs):
        
        '''
        Parameters
        ----------
        
        list_seeds:     list, seq
                        seed sequences for the optimizer

        max_attempts:   int
                        maximum number of mutations per seed
            
        Returns
        -------
        ga_df:  dataframe
                Sequence, Efficacy

        '''
        self.__T = kwargs.get('T', 0.1)
        self.__max_attempts = kwargs.get('max_attempts', 100)

        print('Setting up Optimizer')
        
        self.seq_df = pd.DataFrame(columns=['seed', 'new_dict'])
        
        for counter, seed in enumerate(list_seeds):
            self.seq_df.at[counter, 'seed'] = seed
            self.seq_df.at[counter, 'new_dict'] = {}
            
        new_seq_dict = {}
        
        for self.i in range(self.seq_df.shape[0]):
            print ('Optimizing Seed ', self.i+1)
            print(f'Sequence: {list_seeds[self.i]}')
            self.__genetic_algorithm()
            new_seq_dict.update(self.seq_df.at[self.i, 'new_dict'].items())
            
        print ('Post-Processing Optimized Sequences')
        
        return self.__post_process(new_seq_dict)