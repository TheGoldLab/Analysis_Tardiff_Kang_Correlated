# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:01:23 2023

@author: ntard

This code is for fitting/assessing ddms fit to empricial data.

See ddm_corr.py for functions to for fitting/assessing/sampling simulated ddms for correlated data
"""

import numpy as np
import pandas as pd
import gddmwrapper as gdw
from os import path
from ddm import Model


def fit_load_model(samps,model,fit_dir='./fits',**kwargs):
    
    do_fit = isinstance(model,Model) #check if passed a ddm model or a string
        
    if do_fit:
        fits = {}
        for subji,sampi in samps.items():
            print('Fitting model %s for subject %s' %(model.name,subji))
            fits[subji] = gdw.run_model(samps[subji],model=model,subj=subji,
                                      out_dir=fit_dir,**kwargs)
    else:
        fits = gdw.load_models(path.join(fit_dir,model))
        fits = {m.subject:m for m in fits}
        
    return fits

def get_all_params(fits,verbose=True):
    params = [gdw.get_params(mod,diagnostics=True) for mod in fits.values()]
    params = pd.concat(params)
    params.drop('noise',inplace=True)
    
    if verbose:
        print(params.head())
        print(params.loc[params.hit_boundary])
    
    return params

def get_all_fit_stats(fits,verbose=True,outfile=None):
    fiteval = [gdw.get_fit_stats(mod) for mod in fits.values()]
    fiteval = pd.DataFrame(fiteval)
    
    if verbose:
        print(fiteval.head())
        
    if outfile:
        fiteval.to_csv(outfile,index=False)             
    
    return fiteval

def get_all_sols(samps,fits,verbose=True,**kwargs):
    assert np.equal(fits.keys(),samps.keys()), "Samples and fits don't match!" #this is an UNORDERED comparison!
    soldf = [gdw.get_predicted(f,samps[k],undec=True,forced=True,err_RT=True,**kwargs) 
                for k,f in fits.items()]
    
    soldf = pd.concat(soldf,ignore_index=True)
    
    if verbose:
        print(soldf.head())
        soldf.mean_undec.hist()
        
    return soldf

def data_pred_merge(data,soldf,on,verbose=True):
    #merge predictions w/ data for analysis
    datam1 = data.merge(soldf.drop('mean_undec',axis=1),on=on)
    assert len(data)==len(datam1),'Original data and merged data do not match!'

    #define psychometric curve in terms of proprotion choose right (right is an error for SNR < 0)
    datam1['pred_response'] = datam1['mean_corr']
    datam1.loc[datam1.SNR<0,'pred_response'] = datam1.loc[datam1.SNR<0,'mean_err'] 
    
    if verbose:
        print(datam1.head())

    return datam1

