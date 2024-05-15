from .general import *
import re
from glob import glob
import os

class Dataset:
    '''
    datasets path and subjs
    '''
    def __init__(self, species=None, quiet_mode=False):
        '''
        List all available datasets, only if the subfolder starts with a capital letter
        '''
        if species:
            quiet_mode=True
        self.root_path = os.path.abspath('/mnt/cube/kai/datasets')
        self.root_path_save = os.path.abspath('/mnt/cube/kai/data_processed/datasets')
        self.path_results = os.path.abspath('/mnt/cube/kai/results/sound_texture')
        self.available_species = sorted([subdir for subdir in next(os.walk(self.root_path))[1] if subdir[0].isupper()])
        self.quiet_mode=quiet_mode
        if species:
            self.assign(species)
        else:
            if not self.quiet_mode:
                print(f'Available species: {self.available_species}')
                paragraph_print('Use Dataset.assign() method to assign a species!')
        
    def assign(self, species):
        '''
        Assign species to species of interest(speciesOI)
        '''
        self.species = species
        self.path = os.path.join(self.root_path, self.species)
        if self.species == 'LibriSpeech':
            self.path = os.path.join(self.path, 'test-clean')
        elif self.species == 'TIMIT':
            self.path = os.path.join(self.path, 'TRAIN')
        elif self.species == 'Mouse':
            self.path = os.path.join(self.path, 'all_wavs')
        self.path_save = os.path.join(self.root_path_save, self.species)
        self.subjs = self.list_subjs(self.species)
        
    def summary(self):
        species_summary(self.species)
    
    def find_subj_path(self, subj):
        
        if self.species in ['BengaleseFinch', 'CaThrasher', 'CaVireo']:
            subj_path = os.path.join(self.path, subj, 'wavs')
        elif self.species == 'SwampSparrow':
            raise ValueError('Species not initialized yet!')
        else:
            subj_path = os.path.join(self.path, subj)
        return(subj_path)
            
    def list_subj_wav_files(self, subj, specify_name_pattern=None):
        if subj=='all':
            path = self.path
        else:
            path = self.find_subj_path(subj)
        
        if specify_name_pattern:
            wav_name_pattern = specify_name_pattern
        elif self.species == 'ZebraFinch':
            wav_name_pattern = '*-*'
        elif self.species in ['CaThrasher', 'CaVireo']:
            wav_name_pattern = '*denoised'
            if not self.quiet_mode:
                fancy_print('Only counting denoised wav files.')
        else:
            wav_name_pattern = '*'
        wav_files = find_wav_files(path, wav_name_pattern)
        return(wav_files)
    
    def list_subjs(self, species=None, exclude='\?', include='.'):
        '''
        list available subjects of a specific species
        if species is not given, use speciesOI attribute
        exclude patterns specified ('.' is python's wildcard for *)
        only include patterns specified (default to be everything) 
        '''
        exclude_pattern = re.compile(exclude)
        include_pattern = re.compile(include)
        subjs = [subj for subj in sorted(next(os.walk(self.path))[1]) 
                 if (not re.match(exclude_pattern, subj)) and re.match(include_pattern, subj)]
        if not self.quiet_mode:
            print(f'Available subjects for {self.species} are:')
            paragraph_print(subjs)
        return(subjs)
    
    def find_path(self, *args):
        return(os.path.join(self.path, *args))
    
def find_wav_files(path, name_pattern=None):
    if not name_pattern:
        name_pattern = '*'
    wav_files = sorted(glob(os.path.join(path, '**/'+name_pattern+'.wav'), recursive=True) + 
                      glob(os.path.join(path, '**/'+name_pattern+'.WAV'), recursive=True))
    return wav_files

def species_summary(species):
    species_path = os.path.join('/mnt/cube/kai/datasets', species)
    if species == 'LibriSpeech':
        species_path = os.path.join(species_path, 'test-clean')
    elif species == 'TIMIT':
        species_path = os.path.join(species_path, 'TRAIN')
    elif species == 'Mouse':
        species_path = os.path.join(species_path, 'all_wavs')
    paragraph_print(f'Species raw data located at {species_path}')
    save_path = os.path.join('/mnt/cube/kai/data_processed/datasets', species)
    paragraph_print(f'Species processed data are saved at {save_path}')
    subjs = sorted([subj for subj in next(os.walk(species_path))[1] if subj != '?'])
    print(f'Available subjects are:')
    paragraph_print(subjs)
    subj = subjs[0]
    if species in ['BengaleseFinch', 'CaThrasher', 'CaVireo']:
        subj_path = os.path.join(species_path, subj, 'wavs')
        if not self.quiet_mode:
            paragraph_print(f'NOTE!!! wav location for an example subject: {subj_path}')
    elif species == 'SwampSparrow':
        subj_path = os.path.join(species_path, subj)
        if not self.quiet_mode:
            fancy_print(f'{species} has not been processed!', 'X')
    else:
        subj_path = os.path.join(species_path, subj)
        if not self.quiet_mode:
            paragraph_print(f'Example subject wav location: {subj_path}')
    
    assert os.path.exists(subj_path), 'Subject path does not seem right!!'
    if species == 'ZebraFinch':
        if not self.quiet_mode:
            paragraph_print(f'NOTE FOR ZEBRA FINCH!!! Recordings with only number and - are actual recording, example: 13-25-07-01.wav')