# load the libraries
import numpy as np
import pandas as pd
from main_3 import  graph
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import main_2 as m

fname = 'donnees_aero_starboard.dat'

data = pd.read_csv(fname, sep='\s+', header=0)

colonnes_names = list(data.columns)
dico_colonnes = {col: [] for col in colonnes_names}


for c in colonnes_names:
    dico_colonnes[c] = data[c].tolist()

print(dico_colonnes.keys())

#plot the speed as function of the time
graph(dico_colonnes['Temp'], dico_colonnes['X'], title = 'déplacement selon X')
graph(dico_colonnes['Temp'], dico_colonnes['Vvent'], title = 'déplacement selon X')

