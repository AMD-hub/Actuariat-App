import warnings
import matplotlib.pyplot as plt
import algo.bestEstimate as be
import pandas as pd
import numpy as np
from algo.ChainModels import Triangle

warnings.filterwarnings("ignore")

def tr2df(triangle, cumul=True) :
    if cumul : 
        df = pd.DataFrame(triangle.Cum).assign(
            Années=triangle.years
        ).reindex(columns=['Années'] + list(pd.DataFrame(triangle.Cum).columns))
    else : 
        df = pd.DataFrame(triangle.Inc).assign(
            Années=triangle.years
        ).reindex(columns=['Années'] + list(pd.DataFrame(triangle.Inc).columns))

    return df



def SimulateBE(model_reg,model_charge, tauxZC, num_simulation,act=True, method= "Bootstrap") : 
    simulated_BE = []
    triangle_psap = Triangle(years=model_charge.FullTriangle.years,data= model_charge.FullTriangle.Cum - model_reg.FullTriangle.Cum,isCumul=True,inferior_nan = True)
    for _ in range(num_simulation) :
        triar = model_reg.Simulate(method)
        triach = Triangle(years=triangle_psap.years,data=triar.Cum+triangle_psap.Cum,isCumul=True)
        triac  = model_charge.refit(triach)
        if act :
            be_tot = be.calculate_be_actualise(tr2df(triac,cumul=True), tr2df(triar,cumul=True),tauxZC).iloc[-1,-1]
        else : 
            be_tot = be.calculate_be_tables(tr2df(triac,cumul=True), tr2df(triar,cumul=True))[1].iloc[-1,-1]
        simulated_BE.append(be_tot)
    return simulated_BE

def Plot_RiskAdjustement(simulations,confidence) : 
    quantile_value = np.quantile(simulations, confidence)
    plt.hist(simulations, bins=30, color='skyblue', edgecolor='black', label='BE <= quantile')
    plt.axvline(quantile_value, color='red', linestyle='dashed', linewidth=1, label=f'BE at Risk (α={confidence})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of BE, and RA = {np.quantile(simulations, confidence) - np.mean(simulations):.2f}')
    plt.legend()
    plt.show()

def Calc_RiskAdjustement(simulations,confidence) : 
    return  np.quantile(simulations, confidence) - np.mean(simulations)