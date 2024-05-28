# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:32:57 2022

DDM specifications for: 
"Normative evidence weighting and accumulation in correlated environments" 
Tardiff et al., 2024.

@author: ntard
"""
import numpy as np
from ddm import Model, Fittable,Drift,Bound,Noise
from ddm.models import NoiseConstant, BoundConstant, BoundCollapsingLinear, OverlayChain, OverlayNonDecision, OverlayUniformMixture


# Need a subclass to make the drift rate vary linearly with SNR, from the tutorial (https://pyddm.readthedocs.io/en/latest/quickstart.html)
class DriftSNRCorr(Drift):
    name = 'Drift depends linearly on SNR, separte drift per rho'
    required_parameters = ['driftSNRn','driftSNR0','driftSNRp'] # <-- Parameters we want to include in the model
    required_conditions = ['mu','rho'] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        #import numpy as np
        v_switch = {
            -0.6 : self.driftSNRn,
            0.0 : self.driftSNR0,
            0.6 : self.driftSNRp
        }
        
        drift = v_switch[conditions['rho']]
        
        return drift * np.abs(conditions['mu'])
    
class DriftSNRRvscale(Drift):
    name = 'Drift depends linearly on SNR, separte drift per rho,adjusted by rho-relative sd'
    required_parameters = ['driftSNRn','driftSNR0','driftSNRp'] # <-- Parameters we want to include in the model
    required_conditions = ['mu','rho'] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        v_switch = {
            -1.0 : self.driftSNRn,
            0.0 : self.driftSNR0,
            1.0 : self.driftSNRp
        }
        
        sd_scale = np.sqrt(1+conditions['rho'])
        drift = v_switch[np.sign(conditions['rho'])] / sd_scale
        
        return drift * np.abs(conditions['mu'])
    

class DriftSNRShared(Drift):
    name = 'Drift depends linearly on SNR, adjusted by rho-relative sd'
    required_parameters = ['driftSNR0'] # <-- Parameters we want to include in the model
    required_conditions = ['mu','rho'] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        #import numpy as np
        sd_scale = np.sqrt(1+conditions['rho'])
        drift = self.driftSNR0 / sd_scale
                
        return drift * np.abs(conditions['mu'])
    
    
class DriftSNR(Drift):
    name = 'Drift depends linearly on SNR'
    required_parameters = ['driftSNR'] # <-- Parameters we want to include in the model
    required_conditions = ['mu'] # <-- Task parameters ("conditions"). Should be the same name as in the sample.
    
    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftSNR * np.abs(conditions['mu'])
    
    
class BoundCorr(Bound):
    name = "Correlation-dependent bound"
    required_parameters = ["Bn", "B0", "Bp"]
    required_conditions = ['rho']
    def get_bound(self, conditions, *args, **kwargs):
        
        b_switch = {
            -0.6 : self.Bn,
            0.0 : self.B0,
            0.6 : self.Bp
        }
        
        return b_switch[conditions['rho']]
    
           
class BoundRScaleCollapsingLinearT(Bound):
    name = "Correlation-dependent bound w/ linear collapse, post-collapse normalization"
    required_parameters = ["Rn", "B0", "Rp","t"]
    required_conditions = ['rho']
    def get_bound(self, t, conditions, *args, **kwargs):

        #Trusting you're inputting the right correlations here!!!
        r_switch = {
            -1.0 : self.Rn,
            0.0 : 0.0,
            1.0 : self.Rp
        }
        
        Bt0 = max(self.B0 - self.t*t, 0.)
        
        scale = np.sqrt(1+r_switch[np.sign(conditions['rho'])])/np.sqrt(1+conditions['rho'])
        # scale = np.sqrt(1+r_switch[conditions['rho']])/np.sqrt(1+conditions['rho'])
        return Bt0*scale
    
    
class BoundSharedCollapsingLinearT(Bound):
    name = "Correlation-independent bound w/ linear collapse, adjusted by rho-relative sd"
    required_parameters = ["B0","t"]
    required_conditions = ['rho']
    def get_bound(self, t, conditions, *args, **kwargs):

        Bt0 = max(self.B0 - self.t*t, 0.)
        
        sd_scale = np.sqrt(1+conditions['rho'])
        
        return Bt0/sd_scale

    

#define fisher transforms (not sure these are still needed here but leaving in case used in scripts)
def fisherz(r):
    return 0.5*np.log((1+r)/(1-r))

def fishezr(z):
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)

    
class NoiseCorr(Noise):
    name = "Correlation-dependent noise"
    required_parameters = ["noisep","noise0","noisen"]
    required_conditions = ['rho']
    def get_noise(self, conditions, **kwargs):
        n_switch = {
            -0.6 : self.noisen,
            0.0 : self.noise0,
            0.6 : self.noisep
        }
        
        return n_switch[conditions['rho']]

    
#generates sds used in the simulated model
gSigma = np.sqrt(2)/2
rs = np.array([-0.6,0.0,0.6])

#this scale is generative sd for sum of pair distribution
#scale = 1/(gSigma*np.sqrt(2*(1+rs)))
sds = np.round(gSigma*np.sqrt(2*(1+rs)),8)
#print(sds)

T_dur = 16 #36.1 #14.92 #Ugh some very long RTs in full datset

this_ddm = Model(name='sampling model',
        drift=DriftSNRCorr(driftSNRn=19,driftSNR0=19,driftSNRp=19),
        noise=NoiseCorr(noisen=sds[0],noise0=sds[1],noisep=sds[2]),
        bound=BoundCorr(Bn=1.7,B0=1.7,Bp=1.7),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=.1),
                                        OverlayUniformMixture(umixturecoef=.01)]),
        dx=0.005, dt=0.005, T_dur=T_dur) #Make T_dur just above max RT and make dt = 0.001



#for separate fits to each correlation condition
ddm_base = Model(name='base',
        drift=DriftSNR(driftSNR=Fittable(minval=0.1, maxval=40)),
        noise=NoiseConstant(noise=1),
        bound=BoundConstant(B=Fittable(minval=0.1, maxval=6)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=3)),
                                        OverlayUniformMixture(umixturecoef=Fittable(minval=0.001, maxval=0.3))]),
        dx=0.005, dt=0.005, T_dur=T_dur) #Make T_dur just above max RT and make dt = 0.001

#same as ddm_base, but with addition of a linear collapsing bound
ddm_baseCL = Model(name='baseCL',
        drift=DriftSNR(driftSNR=Fittable(minval=0.1, maxval=40)),
        noise=NoiseConstant(noise=1),
        bound=BoundCollapsingLinear(B=Fittable(minval=0.1, maxval=6),
                                    t=Fittable(minval=0.0,maxval=6)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=3)),
                                        OverlayUniformMixture(umixturecoef=Fittable(minval=0.001, maxval=0.3))]),
        dx=0.005, dt=0.005, T_dur=T_dur)



#fit all correlation conditions. Adjust by rho-dependent variance, but share bound and drift
ddm_rsharedCLT = Model(name='rsharedCLT',
        drift=DriftSNRShared(driftSNR0=Fittable(minval=0,maxval=40)),
        noise=NoiseConstant(noise=1),
        bound=BoundSharedCollapsingLinearT(B0=Fittable(minval=0.1, maxval=6),
                        t=Fittable(minval=0.0,maxval=6)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=3)),
                                        OverlayUniformMixture(umixturecoef=Fittable(minval=0.001, maxval=0.3))]),
        dx=0.005, dt=0.005, T_dur=T_dur) 


#linear collapsing bound adjusted by rho cond
ddm_boundrsCLT_sk = Model(name='bound_rscaleCLT_sk',
        drift=DriftSNRShared(driftSNR0=Fittable(minval=0,maxval=40)),
        noise=NoiseConstant(noise=1),
        bound=BoundRScaleCollapsingLinearT(Rn=Fittable(minval=-0.9, maxval=0.9),
                        B0=Fittable(minval=0.1, maxval=6),
                        Rp=Fittable(minval=-0.9, maxval=0.9),
                        t=Fittable(minval=0.0,maxval=6)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=3)),
                                        OverlayUniformMixture(umixturecoef=Fittable(minval=0.001, maxval=0.3))]),
        dx=0.005, dt=0.005, T_dur=T_dur) #Make T_dur just above max RT and make dt = 0.001

#drift varies by correlation condition, linear collapsing bound does not, save adjusment for rho-dependent sd
ddm_boundsharedCLT_vk = Model(name='bound_sharedCLT_vk',
        drift=DriftSNRRvscale(driftSNRn=Fittable(minval=0,maxval=60),
                              driftSNR0=Fittable(minval=0,maxval=40),
                              driftSNRp=Fittable(minval=0,maxval=40)),
        noise=NoiseConstant(noise=1),
        bound=BoundSharedCollapsingLinearT(B0=Fittable(minval=0.1, maxval=6),
                        t=Fittable(minval=0.0,maxval=6)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=3)),
                                        OverlayUniformMixture(umixturecoef=Fittable(minval=0.001, maxval=0.3))]),
        dx=0.005, dt=0.005, T_dur=T_dur) #Make T_dur just above max RT and make dt = 0.001

#same as ddm_boundrsCLT_sk but drift varies by correlation condition
ddm_boundrsCLT_vk = Model(name='bound_rscaleCLT_vk',
        drift=DriftSNRRvscale(driftSNRn=Fittable(minval=0,maxval=40),
                              driftSNR0=Fittable(minval=0,maxval=40),
                              driftSNRp=Fittable(minval=0,maxval=40)),
        noise=NoiseConstant(noise=1),
        bound=BoundRScaleCollapsingLinearT(Rn=Fittable(minval=-0.9, maxval=0.9),
                        B0=Fittable(minval=0.1, maxval=6),
                        Rp=Fittable(minval=-0.9, maxval=0.9),
                        t=Fittable(minval=0.0,maxval=6)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=3)),
                                        OverlayUniformMixture(umixturecoef=Fittable(minval=0.001, maxval=0.3))]),
        dx=0.005, dt=0.005, T_dur=T_dur) #Make T_dur just above max RT and make dt = 0.001



