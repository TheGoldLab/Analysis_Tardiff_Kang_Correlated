#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 2020


DDM fitting script for:
"Normative evidence weighting and accumulation in correlated environments" 
Tardiff et al., 2024.

    
@author: ntardiff

"""

#import ddm_corr_subj as ddm_corr
import gddmwrapper as gdw
import models_corr
import helpers

import argparse
import math
import copy
import numpy as np
#import pandas as pd
#import scipy.stats as st
from ddm import set_N_cpus,display_model


def main():
    #set constants
    FITS_DIR = './fits'
    FITS_TEST_DIR = './fits_test'
    RTminmax = [0.3,15]
    conds = ['rho','mu']
    
        
    #parse command line arguments
    parser = argparse.ArgumentParser(description='Fit PyDDM models.')
    parser.add_argument('--model_id',action='store',dest='model_id',required=True)
    parser.add_argument('--subject',action='store',dest='subnum',type=int)
    #parser.add_argument('--iteration',action='store',dest='it',type=int)
    parser.add_argument('--session',action='store',dest='sess')
    parser.add_argument('--corr',action='store',dest='corr',type=float,default=None)
    parser.add_argument('--split',action='store',dest='split',type=int,default=None)
    parser.add_argument('--split_N',action='store',dest='split_N',type=int,default=2)
    parser.add_argument('--data',action='store',dest='data')
    parser.add_argument('--parallel',action='store',dest='par',type=int)
    #parser.add_argument('--norestrictpop',action='store_true')
    parser.add_argument('--optfit',action='store',dest='optfit',type=str)
    parser.add_argument('--solmethod',action='store',default=None)
    parser.add_argument('--test',action='store_true')

    args = parser.parse_args()
    #print(args)           
    
    data_file = args.data
    
    if args.par:
        set_N_cpus(args.par)
        
    if args.test:
        OUT_DIR = FITS_TEST_DIR
        load_args = {'verbose': True}
    else:
        OUT_DIR = FITS_DIR
        load_args = dict()
    if args.optfit:
        assert (args.optfit=='faster') or (args.optfit=='finer')
        
        
    #some base arguments we might update later dependng on the model
    it = None
        
    #get model
    model = copy.deepcopy(getattr(models_corr,args.model_id))
    
       
    
    if args.corr is None:
        data_preproc = lambda x: helpers.base_preproc(x,RTminmax)
    else:
        data_preproc = lambda x: helpers.onecorr_preproc(x,args.corr,RTminmax)
        
    if args.split is None:
        data_postproc = None
    else:
        data_postproc = lambda x: helpers.split_postproc(x,args.split,args.split_N)
        
        
    sample,subj = gdw.load_data(data_file,rt='RT',subjnum=args.subnum,
                                    conds=conds,
                                    preproc=data_preproc,
                                    postproc=data_postproc,**load_args)
    
    #gut checks
    #now that we're doing multiple correlation conditions, I've generalized
    #the model code a bit, and I want to make sure that it isn't fed junk.
    #first get correlations in sorted order (should already be sorted...)
    rconds = np.sort(np.asarray(sample.condition_values('rho')))
    #then run checks
    if args.corr is None:
        assert len(rconds)==3, "Only supported for exacty three values of rho."
        assert np.array_equal(np.sign(rconds),[-1.,0.,1.]), "Must have a negative, 0, and a positive rho."
        assert np.abs(rconds[0])==rconds[-1], "Negative and positive rho must be equal in absolute value."
    else:
        assert len(rconds)==1, 'Only one correlation condition requested, multiple or zero returned!'
        
        #also set value of noise based on correlation
        noise = model.get_dependence('Noise')
        assert isinstance(noise,models_corr.NoiseConstant), 'NoiseConstant required for single correlation model!'
        noise_val = np.sqrt(1+rconds[0])
        setattr(noise,'noise',noise_val)
        
    #run model
    print('Fitting model %s for subject %s session %s...' % 
          (model.name,subj,args.sess))
    if args.corr is not None:
        print('Fitting correlation %.1f only...' % args.corr)
    if args.split is not None:
        print('Fitting part %d of %d of data only...' % (args.split,args.split_N))
    print('Data file: %s\n' % data_file)
    if args.optfit or args.solmethod:
        flags = 'Flags: '
        if args.optfit:
            flags+="--optfit %s" % args.optfit
        if args.solmethod:
            flags+="--solmethod %s" % args.solmethod
        print(flags)
    
    if args.test:
        import time
        stime = time.time()
        
    #for more complicated models, it is helpful to keep popsize ~40
    #this is a less exhaustive search of param space w/ only slight impacts on fit
    if args.optfit == 'faster':
        popsize = math.ceil(40./len(model.get_model_parameter_names()))
        fitparams = {'popsize':popsize,'strategy':'rand1bin',
                                         'recombination':0.9}
    elif args.optfit == 'finer':
        fitparams = {'popsize': 18, 'tol':0.0025, 'maxiter': 2000, 
                     'strategy':'rand1bin','recombination': 0.9}
    else:
        fitparams = None
            
                
    
    sess = args.sess if args.corr is None else '%s_o%.1f' % (args.sess,args.corr)
    
    sess = args.sess if args.split is None else '%s_h%d' % (args.sess,args.split)
    
    if args.split and args.split_N!=2:
        sess+= ('-%d' % args.split_N)

    
    model_fit = gdw.run_model(sample,model=model,subj=subj,sess=sess,it=it,
                                  out_dir=OUT_DIR,
                                  fitparams=fitparams,method=args.solmethod)

    if args.test:
        rtime = time.time() - stime
        print('Fitting time: %.2f s (%.2f h)' % (rtime,rtime/60**2))
        
    display_model(model_fit)

   
if __name__=='__main__':
    main()

