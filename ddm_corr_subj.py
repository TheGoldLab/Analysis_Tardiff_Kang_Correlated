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


#separates parameters by component, 
#so can extract all parameters in cases where two params from different components share a name (BAD)
def get_params_by_component(model):
    params = {component.depname:{p : getattr(component,p).real 
               for p in component.required_parameters}
             for component in model.dependencies}

    #params = pd.DataFrame.from_dict(params,orient='index',columns=['value'])
    #params.index.name = 'param'
    
    params2 = pd.DataFrame.from_dict(params, orient="index").stack().to_frame()
    params2.rename({0:'value'},axis=1,inplace=True)
    
    if hasattr(model,'subject'):
        params2.set_index(np.tile(model.subject,len(params2)),append=True,inplace=True)
        #params2['subject'] = model.subject
    
    params2.index=params2.index.set_names(['component','param','subject'])
    
    return params2

def get_all_params_by_comp(fits,verbose=True):
    params = [get_params_by_component(mod) for mod in fits.values()]
    params = pd.concat(params)
    params.drop('Noise',level=0,inplace=True)
    
    if verbose:
        print(params.head())
    
    return params
    

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

#given how we trimmed RTs, renorming unforced better than using forced
def renorm_probs(soldf):
    soldf = soldf.copy()
    soldf[['mean_corr','mean_err']] = \
        soldf[['mean_corr','mean_err']].div(soldf['mean_corr'] + soldf['mean_err'],axis=0)
    return soldf

def get_all_sols(samps,fits,verbose=True,renorm=False,forced=True,undec=True,err_RT=True,**kwargs):
    assert np.equal(fits.keys(),samps.keys()), "Samples and fits don't match!" #this is an UNORDERED comparison!
    #soldf = [gdw.get_predicted(f,samps[k],undec=True,forced=True,err_RT=True,**kwargs)
    soldf = [gdw.get_predicted(f,samps[k],undec,forced,err_RT,**kwargs)
                for k,f in fits.items()]
    
    soldf = pd.concat(soldf,ignore_index=True)
    
    #never renorm if forced is true...already sums to 1
    if (not forced) and renorm:
        print('renorming probs...')
        soldf = renorm_probs(soldf)
    
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

