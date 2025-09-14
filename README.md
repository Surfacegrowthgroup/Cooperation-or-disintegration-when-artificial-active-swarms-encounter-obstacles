# Numerical programes codes and data for "Cooperation and disintgration when artificial swarms encounter obstacles? A direct numerical test"

## Contents

The files contains three Python code files and one data file of figures in paper, the results and data can all be gained by these Python codes.

### Running environment

All of three Python codes work with Python 3.9.12 and we use the following libraries: numpy(1.22.4), matplotlib(3.5.1), scipy(1.13.1), networkx(2.7.1) and tqdm(4.67.0).

### Settings for numerical programs

All the parameters is controled by the class Setup in file "counter_settings.py". The meanings of these control parameters are shown in the table below.

| Parameter    | Meaning                                             |
| ------------ |---------------------                                |
| par_denisty  | the density of active particles $\rho$              |
| que_density  | the density of quenched disorder $\rho_{o}$         |
| times        | the total time steps of one simultaion              |
| St           | the time step that start to take average            |
| loop_times   | the The number of independent calculations          |
| width        | the vertical length of system $W$                   |
| length       | the horizontal length of disorder region $L_{o}$    |
| left_length  | the horizontal length of left display region        |
| place_length | the horizontal length of place region $L_{p}$       |
| white_length | the horizontal length of buffer region $L_{w}$      |
| end_length   | the horizontal length of end region $L_{e}$         |
| radius       | the interactions radius of active particles $R_{a}$ |
| strength     | the amplitude of thermal noise $\eta$               |
| speed        | the speed of active particles $v_{0}$               |
| que_stren    | the amplitude of quenched noise $H$                 |
| que_radius   | the interactions radius of quenched disorder $R_{o}$|

### Drawing of morphology

The display and animation of morphology can be achieved by program "simu_enounter.py".

### Simulation and data computation

The data of figures in paper can be obtained from the program "calcu_encounter.py", the The number of threads in parallel computing can be adjusted by parameters *core_num*.