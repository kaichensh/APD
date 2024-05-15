import numpy as np
import matplotlib.pyplot as plt
import copy
import socket
import os
import sys
import pandas as pd
import scipy.io as sio
import matplotlib
import scipy.signal as sg
import math
import scipy as sp
import pylab
import h5py
import pickle
import seaborn as sns
import random
import collections
import logging
import datetime
import bisect
import itertools
from scipy.io import wavfile
sns.set_palette('colorblind')
import shutil
from glob import glob
import gc
from .general import *

class SoundWav:
    
    def __init__(self, sequence, sr, name):
        self.value = sequence
        self.sr = sr
        self.length = len(sequence)
        self.name = name
    
    #spectrogram function borrowed from zeke, returns [freq, time, spec]
    def get_spec(self, save_par=None, plot=False, save_plot=False, log=True, fft_size=512, step_size=64, window=('gaussian', 80),
                           thresh = 5,
                           f_min=0.,
                           f_max=15000.):

        f, t, specgram = sg.spectrogram(self.value, fs=self.sr, window=window,
                                           nperseg=fft_size,
                                           noverlap=fft_size - step_size,
                                           nfft=None,
                                           detrend='constant',
                                           return_onesided=True,
                                           scaling='density',
                                           axis=-1,
                                           mode='psd')
        if log == True:
            specgram /= specgram.max() # volume normalize to max 1
            specgram = np.log10(specgram) # take log
            specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
        else:
            specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold

        f_filter = np.where((f > f_min) & (f <= f_max))
        if plot:
            self.plot(specgram[f_filter], {'index': 0}, save_plot=save_plot, save_par = save_par)
        else:
            return f[f_filter], t, specgram[f_filter]
    
    def find_noise(self, streak_threshold = 3, threshold_level='high'):
        avgs = []
        avg_step_size=int(self.sr/100)
        win_size = int(self.sr/40)
        for step in np.arange(0, len(self.value), avg_step_size):
            avgs.append(np.mean(abs(self.value[step:step+win_size]))) #mean of every 1000 samples

        if threshold_level == 'high':
            noise_threshold = min(5* np.mean(avgs[:20]), np.mean(avgs)*10)
        elif threshold_level == 'low':
            noise_threshold = min(2* np.mean(avgs[:20]), np.mean(avgs)*1.2)
        else:
            fancy_print('Unknown threshold level!', separation ='X')
        reject_threshold = np.quantile(avgs,  0.5)
        noise_starts = []
        noise_ends = []
        streak_count = 0
        for i, avg in enumerate(avgs):
            seg = self.value[int(i*avg_step_size):int(i*avg_step_size+win_size)]
            seg_max = np.max(np.abs(seg))

            if avg<noise_threshold and seg_max<reject_threshold: 
                streak_count+=1
                if streak_count==streak_threshold:
                    noise_starts.append(int((i-streak_threshold+1)*avg_step_size))
                    noise_ends.append(int(i*avg_step_size+win_size))
                elif streak_count>streak_threshold:
                    noise_ends[-1]=int(i*avg_step_size+win_size)
            else:
                streak_count = 0
        return noise_starts, noise_ends
    def find_onsets_offsets(self, streak_threshold=3, min_len_s=0, avg_step_size=None, 
                            labeled_correction = None, threshold_level='high'):
        '''
        find all the onsets (valley) of the entire sequence 
        streak_threshold: how many consecutive values are required to qualify as a valley
        min_len_s: minimum length (unit: seconds) of the segment required
        avg_step_size: number of samples for each stride during moving average
        '''
        #first calculate moving averate of the signal
        #print(streak_threshold, min_len_s, avg_step_size)
        if not avg_step_size:
            avg_step_size=int(self.sr/100) #100Hz sampling rate for moving average, each sample 10ms
        avgs = []
        win_size = int(self.sr/40)
        for step in np.arange(0, len(self.value), avg_step_size):
            avgs.append(np.mean(abs(self.value[step:step+win_size]))) #mean of every 1000 samples
        #noise_threshold = np.mean(avgs)*1.2
        if threshold_level == 'high':
            noise_threshold = min(5* np.mean(avgs[:20]), np.mean(avgs)*10)
        elif threshold_level == 'low':
            noise_threshold = min(2* np.mean(avgs[:20]), np.mean(avgs)*1.2)
        else:
            fancy_print('Unknown threshold level!', separation ='X')
#         print(f'Noise threshold is {noise_threshold}')
        #then extract all the valleys in the avg plot; onsets are the start of a new peak; offsets are the end of a peak
        onsets = []
        offsets = []
        streak_count = 0
        for i, avg in enumerate(avgs):
            if i>= len(avgs)-min_len_s*self.sr/avg_step_size:
                break
            if avg<noise_threshold: #magnitude here needs to be tuned
#                 if not streak_count and i!=len:
#                     print("offset:"+str(i))
#                     offsets.append(i)
                streak_count+=1
                if streak_count==streak_threshold:
                    #print("offset:"+str(i-streak_threshold+1))
                    offsets.append(int(i-streak_threshold+1))
                    #print('onset:'+str(i))
                    onsets.append(int(i))
                elif streak_count>streak_threshold:
                    #print('onset:'+str(i))
                    onsets[-1]=int(i)
#                     if i==len(avgs)-1:
#                         onsets=onsets[:-1]
                #print(streak_count)
            else:
#                 if streak_count >0 and streak_count<=streak_threshold:
#                     print('erasing previous offset')
#                     offsets = offsets[:-1]
                #print('breaking streak')
                streak_count = 0
#             if streak_count:
#                 print(streak_count)
                
        
        assert len(onsets)==len(offsets), f'{len(onsets)}, {len(offsets)}'
        onsets_abs = [int(onset*avg_step_size+0.5*win_size) for onset in onsets[:-1]]
        offsets_abs = [int(offset*avg_step_size+0.5*win_size) for offset in offsets[1:]]
        
        if labeled_correction:
            #print('Found onsets and offsets, starting correction from labeled data')
            starts, ends = labeled_correction
            onsets_remove = []
            offsets_remove = []
            for onset, offset in zip(onsets_abs, offsets_abs):
                #print(onset, offset)
                if bisect.bisect(starts, onset) == bisect.bisect(ends, onset):
                    if bisect.bisect(starts, onset-2*win_size) == bisect.bisect(ends, onset) or onset>ends[-1]: 
                        # 2000 is a tolerance value for the way I calculate onsets
                        # if an onset is shifted 2000 to the left and still doesnt fall into any start/end range then it means it's not an actual motif  
                        onsets_remove.append(onset)
                        #print(onset)
                        offsets_remove.append(offset)
            #print(onsets_remove)
            for onset, offset in zip(onsets_remove, offsets_remove):
                onsets_abs.remove(onset)
                offsets_abs.remove(offset)
        return onsets_abs, offsets_abs
    
    def find_timestamp_pairs(self, desired_seg_len_s, onsets = None, offsets = None, tolerance_s = 0.5):
        '''
        find timestamp pairs for a desired segment length without cutting off syllables
        desired_seg_len_s: length (unit: seconds) of desired segments where start&end timestamps need to be found
        tolerance_s: length (unit: seconds) before and after each end that can be accepted 
        streak_threshold: how many consecutive values are required to qualify as a valley
        avg_step_size: number of samples for each stride during moving average
        '''
        timestamps = []
        if not onsets or not offsets:
            onsets, offsets = self.find_onsets_offsets(min_len_s = desired_seg_len_s)
        else:
            assert min(onsets)>20, "onsets and offsets need to be in unit of datapoints not seconds"
            
        for onset in onsets:
            closest_offset = offsets[np.argmin([abs(offset-onset-desired_seg_len_s*self.sr) for offset in offsets])]
            if abs(closest_offset-onset-desired_seg_len_s*self.sr)<=tolerance_s*self.sr:
                timestamps.append([onset, closest_offset])
            else:
                continue
        print('Found %d eligible segments in file %s.' %(len(timestamps), self.name))
        return np.array(timestamps)
    
    def find_no_overlap_segs(self, desired_seg_len_s, onsets = None, offsets = None):
        '''
        find onsets and offsets of specific length without overlap
        '''
        if not onsets or not offsets:
            onsets, offsets = self.find_onsets_offsets(min_len_s = desired_seg_len_s)
        else:
            assert min(onsets)>20, "onsets and offsets need to be in unit of datapoints not seconds"
        desired_seg_len = int(desired_seg_len_s*self.sr)
        starts = []
        ends = []
        last_end = 0
        for onset in onsets:
            if onset<last_end: # if onset falls in the last segment
                closest_offset = [offset for offset in offsets if offset<last_end][-1] # closest offset before last end
                if onset<closest_offset:
                    continue
            
            if onset+desired_seg_len>self.length:
#                 print(f'>>Skipping onset {onset} (seg length {desired_seg_len}) near end of recording {self.length}...')
                continue
            starts.append(int(onset))
            ends.append(int(onset+desired_seg_len))
            last_end = ends[-1]
        return starts, ends
    
    def find_fixed_len_bouts(self, desired_seg_len_s, onsets = None, offsets = None, break_threshold_s = 1):
        '''
        find onsets and offsets of specific length without overlap
        here we use 1 seconds as the break threshold for a bout. (Citation below defined bouts as at least 1s apart)
        Eens M. Understanding the complex song of the European starling: an integrated ethological approach, Adv Study Behav, 1997, vol. 26 (pg. 355-434)
        '''
        if not onsets or not offsets:
            onsets, offsets = self.find_onsets_offsets(min_len_s = desired_seg_len_s)
        else:
            assert min(onsets)>20, "onsets and offsets need to be in unit of datapoints not seconds"
        desired_seg_len = int(desired_seg_len_s*self.sr)
        break_threshold = int(break_threshold_s*self.sr)
        starts = []
        ends = []
        bout_offsets = [offsets[i] for i in range(len(onsets)-1) 
                        if (onsets[i+1]-offsets[i])>break_threshold ] + [offsets[-1]]
        bout_onsets = [onsets[0]] + [onsets[i+1] for i in range(len(onsets)-1) 
                                     if (onsets[i+1]-offsets[i])>break_threshold ]
        
#         for onset, offset in zip(bout_onsets, bout_offsets):
#             if offset-onset < desired_seg_len:
#                 continue
#             if onset+desired_seg_len>self.length:
# #                 print(f'>>Skipping onset {onset} (seg length {desired_seg_len}) near end of recording {self.length}...')
#                 continue
#             if offset-onset
#             starts.append(int(onset))
#             ideal_end = onset+desired_seg_len
#             closest_syllable_offset = offsets[np.argmin([abs(syllable_offset-ideal_end) for 
#                                                          syllable_offset in offsets])]
#             if abs(closest_syllable_offset-ideal_end)<0.3*self.sr:  
#                 # if closest syllable offset to ideal end is more than 0.3 seconds apart, then use ideal end instead
#                 chosen_end = closest_syllable_offset
#             else:
#                 chosen_end = ideal_end
#             ends.append(int(chosen_end))
            
        for onset, offset in zip(bout_onsets, bout_offsets):
            seg_onset = onset
            while (offset-seg_onset > desired_seg_len) and (seg_onset+desired_seg_len < self.length):
#                 print(seg_onset, offset)
                starts.append(int(seg_onset))
                ideal_end = seg_onset+desired_seg_len
                closest_syllable_offset = offsets[np.argmin([abs(syllable_offset-ideal_end) for 
                                                             syllable_offset in offsets])]
                if abs(closest_syllable_offset-ideal_end)<0.3*self.sr:  
                    # if closest syllable offset to ideal end is more than 0.3 seconds apart, then use ideal end instead
                    chosen_end = closest_syllable_offset
                else:
                    chosen_end = ideal_end
                ends.append(int(chosen_end))
                for syllable_onset in onsets:
                    if syllable_onset<chosen_end:
                        continue
                    else:
                        seg_onset = syllable_onset
                        break
                if seg_onset == starts[-1]:
                    break
        return starts, ends
    
    def median_syllable_length(self, onsets=None, offsets=None, return_all=False):
        '''
        calculate the mean syllable length using onset offset values
        '''
        if not onsets or not offsets:
            onsets, offsets = self.find_onsets_offsets()
        else:
            assert min(onsets)>20, "onsets and offsets need to be in unit of datapoints not seconds"
            
        lengths = [offset-onset for onset, offset in zip(onsets, offsets)]
        mean = np.median(lengths)
        if not return_all:
            lengths = None
        return mean, lengths
        
    def find_longest_bout(self, onsets=None, offsets=None, break_threshold_s = 3):
        '''
        find the longest bout based on syllable onset and offset values
        if an offset and the next onset are separated for more than the # of sec specified with break_threshold_s,
            a new bout is considered to start
        onsets and offsets are default to be none, in which case default parameters of find_onset_offset will be used
        if onsets and offsets are provided, they need to be in the unit of datapoints instead of seconds
        '''
        if not onsets or not offsets:
            onsets, offsets = self.find_onsets_offsets()
        else:
            assert min(onsets)>20, "onsets and offsets need to be in unit of datapoints not seconds"
            
        last_bout_ends_i = [-1]+list(np.argwhere([(start/self.sr-end/self.sr)>break_threshold_s 
                                    for start, end in zip(onsets[1:], offsets[:-1])]).flatten())+ [len(onsets)-1]
        bout_lengths_pts = [last_bout_ends_i[i+1]-last_bout_ends_i[i] for i in range(len(last_bout_ends_i)-1)]
        longest_bout_i = np.argmax(bout_lengths_pts)
        onset_longest = onsets[last_bout_ends_i[longest_bout_i]+1] 
        offset_longest = offsets[last_bout_ends_i[longest_bout_i+1]]
        
        return onset_longest, offset_longest
    
    def segment_onsets_offsets(self, onsets, offsets, step_size, iterative=False):
        '''
        segment the longest bout into increasingly long segments depending on step_size
        step_size need to be in unit of datapoints (usually multiples of mean syllable length)
        if iterative, the function will iterate until it finds all qualifying bouts (not only the longest bout)
        '''
        onsets_all = copy.deepcopy(onsets)
        offsets_all = copy.deepcopy(offsets)
        onset_longest, offset_longest = self.find_longest_bout(onsets_all, offsets_all)
        num_steps = int(np.floor((offset_longest-onset_longest)/step_size))
        start_step_i = int(np.floor(0.5*self.sr/step_size))
#         print(start_step_i)
        seg_offsets = []
        step_indeces = []
        seg_onsets = []
        wav_names = []
# #         print(onsets_all)
#         print(onset_longest)

        while (num_steps-start_step_i)>=4:
            
#             print(num_steps)
            for i in range(num_steps):
                if (i+1)*step_size<0.5*self.sr:
                    print('Less than 0.5 sec, continue')
                    continue
                ideal_offset = int(onset_longest+(i+1)*step_size)
                closest_offset = offsets_all[np.argmin([abs(offset-ideal_offset) for offset in offsets_all])]
                if abs(closest_offset-ideal_offset)<0.3*step_size:
                    actual_offset = closest_offset
                else:
                    actual_offset = ideal_offset
                if i<16: # max 80 syllable length
                    seg_offsets.append(actual_offset) 
                    step_indeces.append((i+1))
                    seg_onsets.append(onset_longest)
                    num_actual_steps = i-start_step_i+1
            wav_names+=[self.name]*num_actual_steps
            if not iterative:
                break
            del_start = onsets_all.index(onset_longest)
            del_end = offsets_all.index(offset_longest)
#             print(onset_longest, offset_longest)
            del onsets_all[del_start:del_end+1]
            del offsets_all[del_start:del_end+1]
            if not onsets_all:
                break
            onset_longest, offset_longest = self.find_longest_bout(onsets_all, offsets_all)
            num_steps = int(np.floor((offset_longest-onset_longest)/step_size))
#             print(onsets_all)
#             print('*'*100)
        return seg_onsets, seg_offsets, step_indeces, wav_names
    
    def segment(self, timestamps, length = None, save_segs=False, save_par=None, plot=False, save_plot=False):
        '''
        option to segment the entire sequence into smaller chunks
        timestamps can either be a list of start times, or a list of [start, end] tuple
        length is set to be an int or a list. When it's an int, all segments will have a fixed length.
        save_par should have all the info including [name] and [location]
        returned object is a list of segements
        '''
    
        if not (isinstance(timestamps[0], tuple) or length):
            raise ValueError('Either give end timestamps or set lengths.')
        if save_segs and not save_par:
            raise ValueError('Input save folder and recording info for saving the segments.')
        
        num_segs = len(timestamps)
        #when timestamps only consist of start times, make a list called complete_timestamps that has both start times and end times
        if not isinstance(timestamps[0], tuple):
            if not isinstance(length, list):
                complete_timestamps = [(start, start+length) for start in timestamps]
            else:
                assert len(length)==num_segs, "length vector and timestamps vector are of different lengths"
                complete_timestamps = [(timestamps[i], timestamps[i]+length[i]) for i in range(num_segs)]
        else:
            complete_timestamps = timestamps
        
        all_segs = []
        for start, end in complete_timestamps:
            this_seg = self.value[start:end]
            if save_segs:
                file_name = '%s_%d.wav' %(save_par['name'], start)
                file_location = os.path.join(save_par['location'], file_name)
                self.save(file_location, seg = this_seg)
                
            if plot:
                self.plot(list(this_seg), {'index': start}, save_plot=save_plot, save_par = save_par)
                
            all_segs.append(this_seg)
        
        return all_segs
    
    #save wav files with the option of adjusting the amplitude
    def save(self, location, amp_rate=1, seg=None):
        if seg is None:
            seg = self.value
        #waveform_adjusted = np.asarray(seg/max(seg)*amp_rate)
        waveform_adjusted = seg
        scipy.io.wavfile.write(location, self.sr, waveform_adjusted)
    
    @staticmethod
    
    def plot(data, plot_par, save_plot = False, save_par = None):
        plot(data, plot_par, save_plot = save_plot, save_par = save_par)