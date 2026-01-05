#!/usr/bin/env python3

import sys
import toksearch
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore



# Constants and parameters
shot = None
NYOUT_PTS = 101
NGYROS = 11
psin = np.linspace(0, 1.0, NYOUT_PTS)


EOXTARGET = toksearch.PtDataSignal('EOXTARGET').fetch(shot)
EOXBEST = toksearch.PtDataSignal('EOXBEST').fetch(shot)
    
print('Source shot:', shot)
print('Raw size of yout:', EOXBEST['data'].shape)

# Reshape data arrays
EOXBEST['data'] = EOXBEST['data'].reshape(-1, NGYROS, NYOUT_PTS)
print("EOXBEST['data'] shape:", EOXBEST['data'].shape)
EOXTARGET['data'] = EOXTARGET['data'].reshape(-1, NGYROS, NYOUT_PTS)
print("EOXTARGET['data'] shape:", EOXTARGET['data'].shape)



# Get timeslices (master time axis)
timeslices = EOXBEST['times'][:EOXBEST['data'].shape[0]]
print(f"Profile time range: {timeslices[0]} to {timeslices[-1]}")
