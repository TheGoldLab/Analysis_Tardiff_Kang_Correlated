# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:59:38 2022

@author: ntard

This code is for fitting/assessing/sampling simulated ddms for correlated data
See ddm_corr_subj.py for functions to help fit/assess empirical data
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import pickle
import warnings
import glob
import copy
import itertools
from os import path
from datetime import date
from ddm import Sample,Model
from ddm.functions import fit_adjust_model
from gddmwrapper.base import mean_err_time
import models_corr



def set_model_params(model,component,**kwargs):
    component = model.get_dependence(component)
    for param,val in kwargs.items():
        setattr(component,param,val)

def gen_samples(rconds,model,nsamp=100):
    samp_all = []
    for r,v in rconds.items():
        for mu in v:
            m = model.solve({'mu': mu, 'rho':r})
            samp = m.resample(nsamp)
            samp_all.append(samp.to_pandas_dataframe(drop_undecided=True))
            
    samp_all = pd.concat(samp_all)
    
    return samp_all

#more general function for generating samples based on a pandas df of variable combinations.
#as opposed to the fixed rho/mu rconds dict approach of gen_samples.
def gen_samples_df(var_combos,model,nsamp=100):
    samp_all = []
    for _,r in var_combos.iterrows():
        m = model.solve(r.to_dict())
        samp = m.resample(nsamp)
        samp_all.append(samp.to_pandas_dataframe(drop_undecided=True))
            
    samp_all = pd.concat(samp_all)
    
    return samp_all

def preds_to_df(sols,undec=False,forced=False,err_RT=True):
    #helper function to produce a datafram from a set of solution objects
    
    #extract values for each condition and produce dataframe
    soldf = {k:[] for k in list(sols.values())[0].conditions.keys()}
    #soldf.update({'mean_corr':[],'mean_err':[],'mean_undec':[],'mean_RT_corr':[]})
    soldf.update({'mean_corr':[],'mean_err':[],'mean_RT_corr':[]})
    if undec:
        soldf.update({'mean_undec':[]})
    if err_RT:
        soldf.update({'mean_RT_err':[]})
        
    for sol in sols.values():
        for k,v in sol.conditions.items():
            #handle condiiton where we get a numpy number vs. a base type
            if isinstance(v,np.number): 
                soldf[k].append(v.item())
            else:
                soldf[k].append(v)
            
        soldf['mean_RT_corr'].append(sol.mean_decision_time())
        if forced:
            soldf['mean_corr'].append(sol.prob_correct_forced())
            soldf['mean_err'].append(sol.prob_error_forced())
        else:
            soldf['mean_corr'].append(sol.prob_correct())
            soldf['mean_err'].append(sol.prob_error())
        
        if undec:
            soldf['mean_undec'].append(sol.prob_undecided())
            
        if err_RT:
            soldf['mean_RT_err'].append(mean_err_time(sol))

    soldf = pd.DataFrame.from_dict(soldf)
    
    return soldf

#more general function for generating predictions based on a pandas df of variable combinations.
#as opposed to the fixed rho/mu rconds dict approach of gen_predict.
def gen_predict_df(var_combos,model,**kwargs):
    sols = {}
    for _,r in var_combos.iterrows():
        m = model.solve(r.to_dict())
        mkey = frozenset(zip(r.index,r.values))
        sols[mkey] = m
    
    soldf = preds_to_df(sols,**kwargs)
    
    return soldf

#really I shouldn't be solving twice in sample and predict, but going w/ it for now
def gen_predict(rconds,model,**kwargs):
    '''
    gets theoretical predicted correct and errror probabilities and mean RTs for a given
    model (not fitted). Useful for plotting psychometric/chronometric functions.
    Values are based on analytical/numerical solutions to model
    '''
    
    #solve the model for all conditions
    #print('Solving model for all conditions. May take a minute...')
    sols = {}
    for r,v in rconds.items():
        for mu in v:
            m = model.solve({'mu': mu, 'rho':r})
            mkey = frozenset([('mu',mu),('rho',r)])
            sols[mkey] = m

    
    soldf = preds_to_df(sols,**kwargs)
    
    return soldf

def fit_load_model(samp_data,model,rs,rscale,nrun,fit_dir='./fits'):
    do_fit = isinstance(model,Model) #check if passed a ddm model or a string
    fits = []
    
    if do_fit:
        for r in rs:
            for s in rscale:
                for n in range(0,nrun):
                    print('Fitting: rho: %.1f, rscale: %.1f, run: %d' %(r,s,n))
                    this_samp = samp_data.loc[(samp_data.rho==r) & (samp_data.rscale==s) & (samp_data.run==n),:]
                    #print(this_samp.rho.unique())
                    samp_fit = Sample.from_pandas_dataframe(this_samp,
                                      rt_column_name="RT", 
                                      correct_column_name="correct")  
    
                    out_file = 'samp_fit_%s_%.1f_%.1f_%d_%s' % (model.name,r,s,n,date.today())
                    this_fit = fit_adjust_model(sample=samp_fit, model=copy.deepcopy(model), #if you don't copy all fits will point to same object
                                                verbose=False)
                    this_fit.rho = r
                    this_fit.rscale = s
                    this_fit.nrun = n
                    fits.append(this_fit)
    
                    with open(path.join(fit_dir,out_file),'wb') as f:
                            pickle.dump(this_fit,f)
    else:
        for r in rs:
            for s in rscale:
                for n in range(0,nrun):
                    in_file = model % (r,s,n)
                    with open(path.join(fit_dir,in_file),'rb') as f:
                        this_fit=pickle.load(f)
                    fits.append(this_fit)
    return fits

def fit_load_model_full(samp_data,model,rscale,nrun,fit_dir='./fits',
                        var_name='rscale',fit_suffix='',**kwargs):
    #unlike fit_load_model, fits all correlation conditions together
    do_fit = isinstance(model,Model) #check if passed a ddm model or a string
    fits = []
    
    if fit_suffix:
        fit_suffix = '%s_' % fit_suffix
     
    if do_fit:
        for s in rscale:
            for n in range(0,nrun):
                print('Fitting: %s: %.1f, run: %d' %(var_name,s,n))
                this_samp = samp_data.loc[(samp_data[var_name]==s) & (samp_data.run==n),:]
                #print(this_samp.rho.unique())
                samp_fit = Sample.from_pandas_dataframe(this_samp,
                                  rt_column_name="RT", 
                                  correct_column_name="correct")  

                out_file = 'samp_fit_%s_%s_%.1f_%d_%s%s' % (model.name,var_name,s,n,fit_suffix,date.today())
                this_fit = fit_adjust_model(sample=samp_fit, model=copy.deepcopy(model), #if you don't copy all fits will point to same object
                                            verbose=False,**kwargs)
                this_fit.rho = this_samp.rho.unique()
                #this_fit.rscale = s
                setattr(this_fit,var_name,s)
                this_fit.nrun = n
                fits.append(this_fit)

                with open(path.join(fit_dir,out_file),'wb') as f:
                        pickle.dump(this_fit,f)
    else:
        for s in rscale:
            for n in range(0,nrun):
                in_file = model % (s,n)
                file_list = glob.glob(path.join(fit_dir,in_file))
                assert len(file_list) ==1, \
                    'File patterns currently only support returning 1 file!'
                with open(file_list[0],'rb') as f:
                    this_fit=pickle.load(f)
                fits.append(this_fit)
    return fits

def fit_load_model_full_combo(samp_data,model,samp_vars,nrun,
                              fit_dir='./fits',fit_suffix='',strict=True,**kwargs):
    #unlike fit_load_model_full, this allows for subsetting on a combination of multiple 
    #sample variables. samp_vars must be a dict in which the keys are variable names and 
    #the values are lists of the unique variable values.
    do_fit = isinstance(model,Model) #check if passed a ddm model or a string
    fits = []
    
    if fit_suffix:
        fit_suffix = '%s_' % fit_suffix
        
    var_names = list(samp_vars.keys()) #list of sample vars we're subsetting on
    
    var_combos = itertools.product(*samp_vars.values()) #all combos of sample vars
     
    if do_fit:
        for c in var_combos:
            samp_query = ('==%s) & '.join('(' + v for v in var_names) + '==%s)') % c
            #print(samp_query)
            this_combo = samp_data.query(samp_query)
            for n in range(0,nrun):
                fit_msg = ('Fitting: ' + ': %s, '.join(var_names) + ': %s') % c
                fit_msg = (fit_msg + ', run %d') % n
                print(fit_msg)
                
                #now further subset by run
                this_samp = this_combo.loc[this_combo.run==n,:]
                #print(this_samp.shape)
                #print(this_samp)
                #print(this_samp.rho.unique())
                samp_fit = Sample.from_pandas_dataframe(this_samp,
                                  rt_column_name="RT", 
                                  correct_column_name="correct")  
                
                
                vars_out = ('_%.2f_'.join(var_names) + '_%.2f') % c
                out_file = ('samp_fit_%s_' + vars_out + '_%d_%s%s') % (model.name,n,fit_suffix,date.today())
                this_fit = fit_adjust_model(sample=samp_fit, model=copy.deepcopy(model), #if you don't copy all fits will point to same object
                                            verbose=False,**kwargs)
                if 'rho' in this_samp.columns: #in case not using this on correlations
                    this_fit.rho = this_samp.rho.unique()
                for v_name,v in zip(var_names,c):
                    setattr(this_fit,v_name,v)
                this_fit.nrun = n
                fits.append(this_fit)

                with open(path.join(fit_dir,out_file),'wb') as f:
                        pickle.dump(this_fit,f)
    else:
        for c in var_combos:
            for n in range(0,nrun):
                in_file = model % (c + (n,))
                #print(in_file)
                file_list = glob.glob(path.join(fit_dir,in_file))
                #print(file_list)
                assert len(file_list) <= 1, \
                    'File patterns currently only support returning 1 file!'
                
                #if (len(file_list)==0) and strict:
                #    raise OSError(filename=in_file)
                
                try:
                    with open(file_list[0],'rb') as f:
                        this_fit=pickle.load(f)
                    fits.append(this_fit)
                except IndexError:
                    if strict:
                        raise OSError(2,"File not found!",in_file)
                    else:
                        warnings.warn("%s not found!" % in_file)
                    
    return fits


def fit_model_cluster(samp_data,model,samp_vars,nrun,
                              fit_dir='./fits',fit_suffix='',**kwargs):
    if fit_suffix:
        fit_suffix = '%s_' % fit_suffix
        
    var_names = list(samp_vars.keys()) #list of sample vars we're subsetting on
    
    var_combos = itertools.product(*samp_vars.values()) #all combos of sample vars
    
    fits = []
     
    for c in var_combos:
        samp_query = ('==%s) & '.join('(' + v for v in var_names) + '==%s)') % c
        #print(samp_query)
        this_combo = samp_data.query(samp_query)
        for n in nrun:
            fit_msg = ('Fitting: ' + ': %s, '.join(var_names) + ': %s') % c
            fit_msg = (fit_msg + ', run %d') % n
            print(fit_msg)
            
            #now further subset by run
            this_samp = this_combo.loc[this_combo.run==n,:]
            #print(this_samp.shape)
            #print(this_samp)
            #print(this_samp.rho.unique())
            samp_fit = Sample.from_pandas_dataframe(this_samp,
                              rt_column_name="RT", 
                              correct_column_name="correct")  
            
            
            vars_out = ('_%.2f_'.join(var_names) + '_%.2f') % c
            out_file = ('samp_fit_%s_' + vars_out + '_%d_%s%s') % (model.name,n,fit_suffix,date.today())
            this_fit = fit_adjust_model(sample=samp_fit, model=copy.deepcopy(model), #if you don't copy all fits will point to same object
                                        **kwargs)
            this_fit.rho = this_samp.rho.unique()
            for v_name,v in zip(var_names,c):
                setattr(this_fit,v_name,v)
            this_fit.nrun = n
            
            fits.append(this_fit)
            with open(path.join(fit_dir,out_file),'wb') as f:
                    pickle.dump(this_fit,f)
                    
    return fits

def main():
    #set_N_cpus(4)
    
    fit_together = True
    shared_mu = True
    nsamp = 100
    
    rconds = {
        -0.6: [0.01073, 0.01517, 0.02146],
        0.0: [0.01697, 0.02399, 0.03393],
        0.6: [0.02146, 0.03035, 0.04292],
        }
    
    if shared_mu:
        rconds[-0.6] = rconds[0.0]
        rconds[0.6] = rconds[0.0]
        #print(rconds)
    
    fitting_model = models_corr.this_ddm
    
    samp_all = gen_samples(rconds,fitting_model,nsamp)
    
    pred_all = gen_predict(rconds,fitting_model)

    
    #print(samp_all.RT.max())
    
    if fit_together:
    
        model = models_corr.ddm_boundrsFish #models_corr.this_ddm_fit #models_corr.ddm_sharek  #models_corr.ddm_sharek_n0
        
        samp_fit = Sample.from_pandas_dataframe(samp_all,
                                        rt_column_name="RT", 
                                        correct_column_name="correct")
        
        out_file = 'sample_test_fit_%s' % model.name
        if shared_mu: 
            out_file = out_file + '_smu'
        
        this_priors = {'R0':st.norm(loc=0,scale=0.5).logpdf}
        #this_fit = fit_adjust_model(sample=samp_fit, model=model, verbose=True)
        this_fit = fit_adjust_model(sample=samp_fit, model=model, 
                                    lossfunction=models_corr.make_lleprior(this_priors),
                                    verbose=True)
        
    
        with open(out_file,'wb') as f:
                pickle.dump(this_fit,f)
    else:
        model = models_corr.this_ddm_base
        for r in rconds.keys():
            this_samp = samp_all.loc[samp_all.rho==r,:]
            #print(this_samp.rho.unique())
            samp_fit = Sample.from_pandas_dataframe(this_samp,
                              rt_column_name="RT", 
                              correct_column_name="correct")  
            
            out_file = 'sample_test_fit_%s_%.1f' % (model.name,r)
            if shared_mu: 
                out_file = out_file + '_smu'
            
            this_fit = fit_adjust_model(sample=samp_fit, model=model, verbose=True)
            
        
            with open(out_file,'wb') as f:
                    pickle.dump(this_fit,f)
        
            
if __name__ == '__main__':
    main()
    

