#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:46:36 2023

@author: nathan
"""
import numpy as np

#base preprocessing for all data
def base_preproc(data,RTminmax=None):
    #there shouldn't be any other missing data in other columns,
    assert not data.isnull().values.any(), "Missing data detected!"
    
    data.rename(columns={'Unnamed: 0':'trial','r':'rho','MeanMain':'mu','participant':'subject'},inplace=True)
    data.drop('index',axis=1,inplace=True)
    
    if RTminmax:
        data = data.loc[(data.rt >= RTminmax[0]) & (data.rt <= RTminmax[1])]

    return data


#preprocessing for models that fit a single correlation condition
def onecorr_preproc(data,rho,RTminmax=None):
    data = base_preproc(data,RTminmax)
    
    assert rho in data.rho.values, "Requested correlaton not found in data!"
    
    data = data.loc[data.rho==rho]
    
    return data


def split_postproc(data,split,N=2):
    #if requested, split data
    #defaults to halfs. As currently structured, lets all of the remainder go to 
    #the last chunk for N=2. Shouldn't make a big difference in our use cases.
    split_idx = len(data)/N
    splits = np.int64(np.arange(0,N+1)*split_idx)
    data = data.iloc[splits[split-1]:splits[split]]

    return data
    

# def split_postproc(data,split):
#     #if requested, split data in half
#     split_idx = len(data)//2
#     if split==1:
#         data = data.iloc[0:split_idx]
#     elif split==2:
#         data = data.iloc[split_idx:]
#     else:
#         raise ValueError('Split can only be 1 (first half) or 2 (second half)!')

#     return data


