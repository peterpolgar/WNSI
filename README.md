# WNSI
Wireless Network simulator with Swarm Intelligence

The purpose of this software to provide an opportonity to test various swarm intelligence (or other) based routing algorithms on Wireless Sensor Networks (WSN).

A WSN can be parameterized with the so called "global" parameters. A routing algorithm can have several so called "specific" parameters.

A routing algorihm must be a separate modul file, whose requirements can be found in the "modul_specification.txt" file. New moduls can be added by placing them to the "si_algs" subfolder. New moduls, which are already in the "si_algs" subfolder will be loaded at program startup, and new moduls can be loaded into the running appication by placing the modul file into the subfolder and then click on the "R" (refresh) button in the application.

Currently there are five built-in routing algorithm modul which can be used: four Ant Colony Optimization (ACO) based, and a not SI based algorithm: Dijkstra

Currently the Optimization button and the Batch run option do not do anything, these are placeholders for new features.

Built with the Qt framework. The callout chart example of Qt (python variant of this: https://doc.qt.io/qt-5/qtcharts-callout-example.html, which is in the Qt Python package "pyside2") was used, and the WNSI software includes a modified version of it.

Acknowledgement to emulbreh for the implemented Bridson algorithm: https://github.com/emulbreh/bridson/blob/master/bridson/__init__.py

## How to run the application
Prerequisites:

- install MPI4PY: python -m pip install mpi4py

- install Qt for Python: https://doc.qt.io/qtforpython-5/gettingstarted.html

- install a software which is based on the Message Passing Interface (MPI) standard, for example OpenMPI or Microsoft MPI.

If you downloaded the source files, you can run the application from the command line with this command:

mpiexec -n 1 python dm_program_v12.py
