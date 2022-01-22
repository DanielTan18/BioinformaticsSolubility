###################
# import libraries
##################

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors

###################
# custom functions
##################
##calculating molecular descriptors

def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = []
    for i in aromatic_atoms:
        if i==True:
            aa_count.append(1)
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom/HeavyAtom
    return AR

def generate(smiles,verbose=False):

    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1,1)
    i=0
    for mol in moldata:
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData,row])
        i+=1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

###################
# page title
##################

image = Image.open('solubility-logo.jpg')

st.image(image,use_column_width=True)
st.write("""
#Molecular Solubility Prediction Wrb app

This app predicts the **Solubility (LogS)** values of the molecules!

Data obtained from https://pubs.acs.org/doi/10.1021/ci034243x
""")

######################
# Input molecules (Side Panel)
######################

st.sidebar.header('User Input Features')

## read smiles input

SMILES_input = "NCCCC\nCCC\nCN\CC(=O)NC1=CC=C(C=C1)O"

SMILES = st.sidebar.text_area("SMILES input",SMILES_input)
SMILES = "C\n" + SMILES #adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:] #skipes the dummy first item

##calculate molecular descriptors
st.header('Computed molecular descriptors')
X = generate(SMILES)
X[1:] #skips dummy first item

################
# pre-built model
###############

#reads in saved model
load_model = pickle.load(open('solubility_model.pkl','rb'))

#apply model to make predictions
prediction = load_model.predict(X)
#prediction_proba = load_model.predict_proba(X)

st.header('Predicted LogS values')
prediction[1:] #skips dummy first item
