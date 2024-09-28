import warnings
import matplotlib.pyplot as plt
import algo.bestEstimate as be
import pandas as pd
import numpy as np

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

def SimulateBE(model_reg,model_charge, tauxZC, num_simulation,act=True, charge=True) : 
    simulated_BE = []
    for _ in range(num_simulation) :
        triac = model_charge.Simulate()
        triar = model_reg.Simulate()
        if act :
            be_tot = be.calculate_be_actualise(tr2df(triac,cumul=True), tr2df(triar,cumul=True),tauxZC).iloc[-1,-1]
        else : 
            be_tot = be.calculate_be_tables(tr2df(triac,cumul=True), tr2df(triar,cumul=True))[1].iloc[-1,-1]
        simulated_BE.append(be_tot)
    return simulated_BE


def Plot_RiskAdjustement(simulations,confidence) : 
    quantile_value = np.quantile(simulations, confidence)
    plt.hist(simulations, bins=30, color='skyblue', edgecolor='black', label='x <= quantile')
    plt.axvline(quantile_value, color='red', linestyle='dashed', linewidth=1, label=f'BE ar Risk (α={confidence})')
    plt.text(2, 2, f'Risk Adjustement: {np.quantile(simulations, confidence) - np.mean(simulations):.2f}', color='black', 
        horizontalalignment='left', verticalalignment='top')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram with Highlighted Quantile')
    plt.legend()
    plt.show()

def Calc_RiskAdjustement(simulations,confidence) : 
    return  np.quantile(simulations, confidence) - np.mean(simulations)