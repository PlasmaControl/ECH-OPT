#!/usr/bin/env python3

# Run on iris:
# module purge
# module load toksearch
# python plot_state_vector.py
# 
# If needed, run pip install pyqtgraph

import sys
import toksearch
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore



# Constants and parameters
shot = 205796
NYOUT_PTS = 101
NGYROS = 11
PLOT_GYROS = [4, 5, 8, 9, 11]  # Gyros to plot individually (1-based indexing)
psin = np.linspace(0, 1.0, NYOUT_PTS)


try:
    EOXTARGET = toksearch.PtDataSignal('EOXTARGET').fetch(shot)
except:
    # Dummy target if not found
    EOXTARGET = {'data': np.ones((20, NYOUT_PTS)) * 1e6,
                 'times': np.linspace(0, 1, 20)}  
    for i in range(20):
        EOXTARGET['data'][i,:] *= (np.random.randn(1) + 0.5)

try:
    EOXBEST = toksearch.PtDataSignal('EOXBEST').fetch(shot)
except:
    # Dummy best if not found
    EOXBEST = {'data':  1.1*np.ones((20, NGYROS, NYOUT_PTS)) * 1e5,
               'times': np.linspace(0, 1, 20)}  
    EOXBEST['data'] += np.random.randn(20, NGYROS, NYOUT_PTS) * 1e4
    
print('Source shot:', shot)
print('Raw size of yout:', EOXBEST['data'].shape)

# Reshape data arrays
EOXBEST['data'] = EOXBEST['data'].reshape(-1, NGYROS, NYOUT_PTS)
print("EOXBEST['data'] shape:", EOXBEST['data'].shape)
EOXTARGET['data'] = EOXTARGET['data'].reshape(-1, NYOUT_PTS)
print("EOXTARGET['data'] shape:", EOXTARGET['data'].shape)

# Get timeslices (master time axis)
timeslices = EOXBEST['times'][:EOXBEST['data'].shape[0]]
print(f"Profile time range: {timeslices[0]} to {timeslices[-1]}")

# Calculate total profile (sum of all gyrotrons)
total_profile = np.zeros((len(timeslices), NYOUT_PTS))
for igyro in range(NGYROS):
    total_profile += EOXBEST['data'][:, igyro, :]


# Create PyQtGraph application
app = QtWidgets.QApplication(sys.argv)

# Set white background
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Create main window
win = QtWidgets.QWidget()
win.setWindowTitle('ECH Profile Viewer')
win.resize(1200, 800)

# Create layout
layout = QtWidgets.QVBoxLayout()
win.setLayout(layout)

# Create graphics layout for plots
graphics_layout = pg.GraphicsLayoutWidget()
layout.addWidget(graphics_layout)

# Create first plot: Total profile vs Target
plot1 = graphics_layout.addPlot(row=0, col=0, title="Total Profile vs Target")
plot1.setLabel('left', 'Power Density')
plot1.setLabel('bottom', 'Normalized Psi')
plot1.setYRange(0, 2)
plot1.addLegend()

# Curves for plot1
curve_total = plot1.plot(pen=pg.mkPen('b', width=2), name='Total Profile')
curve_target = plot1.plot(pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine), name='Target')

# Create second plot: Individual gyrotron contributions
plot2 = graphics_layout.addPlot(row=1, col=0, title="Individual Gyrotron Contributions")
plot2.setLabel('left', 'Power Density')
plot2.setLabel('bottom', 'Normalized Psi')
plot2.setYRange(0, 1)
plot2.addLegend()

# Create curves for each gyrotron
gyro_curves = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']
for igyro in PLOT_GYROS:
    curve = plot2.plot(pen=pg.mkPen(colors[(igyro-1) % len(colors)], width=1.5), 
                       name=f'Gyro {igyro}')
    gyro_curves.append(curve)

# Create slider widget
slider_widget = QtWidgets.QWidget()
slider_layout = QtWidgets.QHBoxLayout()
slider_widget.setLayout(slider_layout)
layout.addWidget(slider_widget)

# Time label
time_label = QtWidgets.QLabel(f"Time: {timeslices[0]:.4f} s")
slider_layout.addWidget(time_label)

# Slider
slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(len(timeslices) - 1)
slider.setValue(0)
slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
slider.setTickInterval(max(1, len(timeslices) // 20))
slider_layout.addWidget(slider)

# Index label
index_label = QtWidgets.QLabel(f"Index: 0 / {len(timeslices)-1}")
slider_layout.addWidget(index_label)


def update_plots(idx):
    """Update plots based on slider value"""
    # Update plot 1: total vs target
    curve_total.setData(psin, total_profile[idx, :] / 1e6)
    curve_target.setData(psin, EOXTARGET['data'][idx, :] / 1e6)
    
    # Update plot 2: individual gyrotron contributions
    for i, igyro in enumerate(PLOT_GYROS):
        gyro_curves[i].setData(psin, EOXBEST['data'][idx, igyro-1, :] / 1e6)
    
    # Update labels
    time_label.setText(f"Time: {timeslices[idx]:.4f} s")
    index_label.setText(f"Index: {idx} / {len(timeslices)-1}")


# Connect slider to update function
slider.valueChanged.connect(update_plots)

# Initialize plots with first timeslice
update_plots(0)

# Show window and run application
win.show()
sys.exit(app.exec_())

