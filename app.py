import streamlit as st
import pandas as pd
import numpy as np 

from algo.triangleCumule  import execute, mettre_nan_sous_deuxieme_diagonale
from algo.ChainModels import Triangle,ChainMack,ChainLondon
from algo.bestEstimate import calculate_be_actualise,calculate_be_tables
from algo.riskAdjustment import execute as RA

# Set the page config
st.set_page_config(page_title="Reserving Application", layout="wide")

# Function to parse uploaded file content
def parse_contents(uploaded_file):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                return None
            return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
# Function to transform triange to data frame : 
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

# Sidebar for file uploads and settings
st.sidebar.title("Reserving Application")
st.sidebar.image("assets/frs.png", width=300)
uploaded_reglement   = st.sidebar.file_uploader("Upload data: reglement", type=['csv', 'xls', 'xlsx'])
uploaded_psap        = st.sidebar.file_uploader("Upload data: PSAP", type=['csv', 'xls', 'xlsx'])
uploaded_courbe_taux = st.sidebar.file_uploader("Upload data: courbe de taux", type=['csv', 'xls', 'xlsx'])

st.sidebar.header("Settings")
method = st.sidebar.selectbox(
    "Choose method",
    options=["Mack Chain Ladder", "Bootstrap", "London Chain"]
)


if st.sidebar.button("Do Calculation"):
    data_reg = pd.DataFrame({})
    data_charge = pd.DataFrame({})

# --------- Defining Data ----------------------------
    if uploaded_reglement is None or uploaded_psap is None : 
        st.warning("Please upload the reglement data and PSAP data.")
    else : 
        reglement_df = parse_contents(uploaded_reglement)
        reg_tr = Triangle(years=np.array(reglement_df.iloc[:,0]),data=np.array(reglement_df.iloc[:,1:]),isCumul=False)

        psap_df = parse_contents(uploaded_psap)
        psap_tr = Triangle(years=np.array(psap_df.iloc[:,0]),data=np.array(psap_df.iloc[:,1:]),isCumul=True)
        charge_tr = Triangle(years=reg_tr.years,data=reg_tr.Cum+psap_tr.Cum,isCumul=True)

        # Making the data Global variables
        st.session_state.reg_tr    = reg_tr 
        st.session_state.charge_tr = charge_tr 

# --------- Fitting Model ----------------------------
        if method == 'Mack Chain Ladder' : 
            # For reglement triangle : 
            model_reg = ChainMack() 
            model_reg.fit(st.session_state.reg_tr)
            st.session_state.model_reg = model_reg 
            # For charge triangle    :
            model_charge = ChainMack() 
            model_charge.fit(st.session_state.charge_tr)
            st.session_state.model_charge = model_charge 
        elif method == 'London Chain' : 
            # For reglement triangle : 
            model_reg = ChainLondon() 
            model_reg.fit(st.session_state.reg_tr)
            st.session_state.model_reg = model_reg 
            # For charge triangle    :
            model_charge = ChainLondon() 
            model_charge.fit(st.session_state.charge_tr)
            st.session_state.model_charge = model_charge 

        elif method == 'Bootstrap' : 
            st.info("Still under construction")

# --------- Yield Curve   -----------------------------
        taux_df = parse_contents(uploaded_courbe_taux) 
        st.session_state.taux = taux_df

# --------- Best Estimate -----------------------------
        Ftriangle_reg    = tr2df(st.session_state.model_reg.FullTriangle)
        Ftriangle_charge = tr2df(st.session_state.model_charge.FullTriangle)

        if uploaded_courbe_taux is None:
            be_reg, be_charge = calculate_be_tables(Ftriangle_charge, Ftriangle_reg)
        else:
            be_charge = calculate_be_actualise(Ftriangle_charge, Ftriangle_reg,st.session_state.taux)
            be_reg    = be_charge.copy() 

        st.session_state.be_reg = be_reg
        st.session_state.be_charge = be_charge

# Create a layout with columns for buttons and data
col1, col2, col3 = st.columns([1, 1, 1])

# Use the columns to display buttons
with col1:
    chosen_output = st.selectbox(
    "Choose Data to show : ",
    options=["Inputed Data", "Model", "Bilan"]
)

with col2: 
    if chosen_output == "Inputed Data" :
        chosen_data = st.selectbox(
        "Choose Data to show : ",
        options=["Reglement Increments", "PSAP data", "Cumulative Reglement", "Charge data", "Yield Curve"]   )
    
    elif chosen_output == "Model" : 
        chosen_data = st.selectbox(
        "Choose Data to apply model : ",
        options=["Reglement","Charge"]) 

        chosen_thing = st.selectbox(
        "Choose Data to show : ",
        options=["Model Parameters","Fitted Triangle [cumulative]", "Fitted Triangle [increments]", "Reserves"]   )

    elif chosen_output == "Bilan" : 
        chosen_data = st.selectbox(
        "Choose Componant to show : ",
        options=["Best Estimate","Total Bilan"]   )

with col3: 
    goo = st.button("Goooo")


if goo:
    if chosen_output == "Inputed Data" : 
        if chosen_data=="Reglement Increments" : 
            st.dataframe(tr2df(st.session_state.reg_tr,cumul=False))
        elif chosen_data=="PSAP data" : 
            st.dataframe(parse_contents(uploaded_psap))
        elif chosen_data=="Cumulative Reglement":
            st.dataframe(tr2df(st.session_state.reg_tr,cumul=True)) 
        elif chosen_data=="Charge data" :
            st.dataframe(tr2df(st.session_state.charge_tr,cumul=True)) 
        elif chosen_data=="Yield Curve" : 
            st.dataframe(st.session_state.taux)
        else :
            st.warning("Plz Choose Data to Show !!")

    elif chosen_output == "Model" : 
        if chosen_data=="Reglement":
            model = st.session_state.model_reg 
        else : 
            model = st.session_state.model_charge 

        if chosen_thing=="Model Parameters":
            if method == 'Mack Chain Ladder':
                summary_fitting = pd.DataFrame({
                    'Devs': [i for i in range(model.DevFactors.shape[0])],
                    'Devs Factors': model.DevFactors,
                    'Devs Deviations': model.Deviations
                })
            elif method == 'London Chain' : 
                summary_fitting = pd.DataFrame({
                    'Devs': [i for i in range(model.Intercepts.shape[0])],
                    'Devs Intercepts': model.Intercepts,
                    'Devs Slopes': model.Slopes
                })
            elif method == 'Bootstrap' : 
                st.info("Still under construction")
            
            st.dataframe(summary_fitting)

        elif chosen_thing == "Fitted Triangle [cumulative]":
            st.dataframe(tr2df(model.FullTriangle,cumul=True)) 
        elif chosen_thing == "Fitted Triangle [increments]":
            st.dataframe(tr2df(model.FullTriangle,cumul=False)) 
        elif chosen_thing == "Reserves":
            st.dataframe(model.Provisions())
        else :
            st.warning("Plz Choose Data to Show !!")
    elif chosen_output == "Bilan" : 
        if chosen_data == "Best Estimate" : 
            st.dataframe(st.session_state.be_charge)
        else : 
            if method == "Bootstrap" :
                Ftriangle_reg    = tr2df(st.session_state.model_reg.FullTriangle)
                Ftriangle_charge = tr2df(st.session_state.model_charge.FullTriangle)
                alpha = st.slider('Select Confidence:', min_value=0.7, max_value=1.0, value=0.8, step=0.01)
                ra = RA(Ftriangle_charge, Ftriangle_reg,st.session_state.taux,alpha)
                df = pd.DataFrame({
                            "BE" :  [ra[2]],
                            "RA" :  [ra[-1]], 
                            "CSM" : [-ra[2] - ra[-1]]
                        })
                st.dataframe(df)
            else :
                st.warning("Risk adjustment is only working with Bootstrap method")  
    else : 
        st.warning("Plz Choose Output to Show !!")


# python -m streamlit run app.py