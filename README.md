# Pele_PP
A collection of my working code to Post-Process PeleLMeX files

## Code Hierarchy
The repository is separated into 3 directories:
- custom_PA: Customized code of PeleAnalysis (not much in common anymore) to post-process AMReX plotfiles into "reduced data structures" (2D slice, statistics tables, modified plotfiles, ...)
- python_PP: python code to read and work with the data structures created with custom_PeleAnalysis more in details (generate images, compute quantities
- dataset_Bunsen_3D: just a plotfile from of a working simulation in PeleLMeX Production directory to test the codes
Code directories are separated in "Src", containing the source code (user doesn't need to go in there) and "Exec" which is where user files are

## Quick setup
First, clone the repository to tour device:
```bash
git clone https://github.com/Izayae/Pele_PP.git
```
It will ask you to connect to your github account. Connecting with the usual password does not work, you need to create a Personal Access Token (PAT) and put it in place of your password. for this procedure, please look at https://mgimond.github.io/Colby-summer-git-workshop-2021/authenticating-with-github.html

## Run examples
### custom_PA
Go to Exec directory and slice_2D. Modify the GNUmakefile by updating the location of your PeleLMeX directory on your device.
If you're on niagara, you can then run:
```bash
source slice_2d.sh
```
Else, modify the bash filed "slice_2d.sh" to your need.
"slice_2d.sh" gives an example on how to automatically detect all available plotfiles, and run the code in a loop on all or a subset of them.

The same steps can be applied with the stats directory.

Note that customize the bash files however you need afterward for your case

### python_PP
Once the intermediary data structures are generated by custom_PeleAnalysis, open the jupyter notebook in python_PP/Exec.
It provides some code examples to produce useful figures. You should be able to run all the cells directly.

You can then create your own python code from there for the figures of your dreams !

## This is a very crude code so if you have issues/recommendations/requests, please contact me :)
