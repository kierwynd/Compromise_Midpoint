# Compromise_Midpoint
Contains the code and data for "Facilitating Compromise in Redistricting with Transfer Distance Midpoints" paper.

## Data
The **GridShapefiles** folder contains several shapefiles for square grids. The **Grid_Plans** folder contains assignment files for several horizontal (A) and vertical (B) stripe plans. The **RandomPartitions_Grids** folder contains several randomly generated grid plans on 24x24 and 40x40 grid graphs with 4 and 8 parts (used for paper experiments).

The **MO_Tracts_projected32615_Data_Plan** folder contains the 2020 census tract shapefiles (including tract population) for Missouri. The **RandomPartitions_MO** folder contains several randomly generated Missouri congressional district plans (used for paper experiments).

## Code

The **Environment_Opt_Windows.yml** environment file details the versions of all packages that the code uses.

The **distance.py** code contains a function to calculate the transfer distance between two given partitions.

The **midpoint.py** code contains functions to determine a midpoint (or sequence of fractional points) between two given partitions.

The **fliplocalsearch.py** code contains local search functions that can be used to generate a heuristic warm-start solution for the MIP in midpoint.py.

The **Examples.ipynb** code contains example calls to the functions in midpoint.py using the data listed above.
