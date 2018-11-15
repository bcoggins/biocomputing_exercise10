# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:59:55 2018

@author: bretl
"""
###Question 1
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from plotnine import *

data=pandas.read_csv("data.txt",header=0,sep=',')
a=ggplot(data, aes(x='x', y='y' ))+theme_classic()+xlab('x')+ylab('y')
a+geom_point()
x=data.x
y=data.y


def lin(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

def quad(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*obs.x**2
    nll1=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll1



linGuess=numpy.array([1,1,1])
linfit=minimize(lin,linGuess,method="Nelder-Mead",options={'disp': True},args=data)
print(linfit.x)

quadGuess=numpy.array([1,1,1,1])
quadfit=minimize(quad,quadGuess,method="Nelder-Mead",options={'disp': True, "maxiter" : 100000},args=data)
print(quadfit.x)


teststat=2*(quadfit.fun-linfit.fun)

df=len(quadfit.x)-len(linfit.x)

1-chi2.cdf(teststat,df)


### Question 2
import scipy
import scipy.integrate as spint
def popSim(y,t0,RN1,KN1,aN2N1,RN2,KN2,aN1N2):
    N1=y[0]
    N2=y[1]
    
    dN1dt=RN1*(1-(N1+aN2N1*N2)/KN1)*N1
    dN2dt=RN2*(1-(N2+aN1N2*N1)/KN2)*N2
    
    return [dN1dt,dN2dt]

# case 1 (both with intraspecific competition <interspecific competition --same parameters)
times=range(0,500)
y0=[1,1]
params=(0.5,20,0.5,0.5,20,0.5)
sim=spint.odeint(func=popSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"pop1":sim[:,0],"pop2":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop1"))+geom_line(color='blue')+geom_line(simDF,aes(x="t",y="pop2"),color='red')+theme_classic()+ylab("population size")

# case 2 (pop1 has higher interspecific competition)
times=range(0,500)
y0=[1,1]
params=(0.5,20,2,0.5,20,0.5)
sim=spint.odeint(func=popSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"pop1":sim[:,0],"pop2":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop1"))+geom_line(color='blue')+geom_line(simDF,aes(x="t",y="pop2"),color='red')+theme_classic()+ylab("population size")

# case 3 (pop1 has less intraspecific competition)
times=range(0,500)
y0=[0.1,0.1]
params=(0.5,50,2,0.5,20,2)
sim=spint.odeint(func=popSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"pop1":sim[:,0],"pop2":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop1"))+geom_line(color='blue')+geom_line(simDF,aes(x="t",y="pop2"),color='red')+theme_classic()+ylab("population size")
