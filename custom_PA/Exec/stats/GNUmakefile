# Path to PeleLMeX (Change to your case)
TOP = /project/b/bsavard/mavab/
PELELMEX_HOME      = ${TOP}/PeleLMeX
PELE_PHYSICS_HOME  = ${PELELMEX_HOME}/Submodules/PelePhysics
AMREX_HOME         = ${PELE_PHYSICS_HOME}/Submodules/amrex
AMREX_HYDRO_HOME   = ${PELELMEX_HOME}/Submodules/AMReX-Hydro

# Path to Mar_PeleAnalysis (Change to your case)
PELE_ANALYSIS_HOME = ../../../custom_PeleAnalysis

# AMReX
DEBUG	      = FALSE
DIM	          = 3
PRECISION     = DOUBLE

# Compilation
COMP          = gnu
USE_MPI       = TRUE
USE_OMP       = FALSE
USE_CUDA      = FALSE

# Executable (read in bash script)
#EBASE = stats

NEEDS_f90_SRC = FALSE
#NEEDS_f90_SRC = TRUE

Pdirs   := Base Boundary AmrCore Extern/amrdata LinearSolvers/MLMG
Ppack   += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

PPdirs  := Source/Utility/PltFileManager 
Ppack += $(foreach dir, $(PPdirs), $(PELE_PHYSICS_HOME)/$(dir)/Make.package)
Blocs += $(foreach dir, $(PPdirs), $(PELE_PHYSICS_HOME)/$(dir))

PAdirs  := Src
Ppack += $(foreach dir, $(PAdirs), $(PELE_ANALYSIS_HOME)/$(dir)/Make.package)
Blocs += $(foreach dir, $(PAdirs), $(PELE_ANALYSIS_HOME)/$(dir))

ifeq ($(EBASE),partStream)
  USE_PARTICLES = TRUE
  Pdirs += Particle
  CEXE_headers += StreamData.H    StreamPC.H
  CEXE_sources += StreamData.cpp  StreamPC.cpp
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

CEXE_sources += $(EBASE).cpp
ifeq ($(NEEDS_f90_SRC),TRUE)
  f90EXE_sources += $(EBASE)_nd.f90
endif
INCLUDE_LOCATIONS += .
VPATH_LOCATIONS   += .

include $(Ppack)
INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

#SDF_LOC = $(PELE_ANALYSIS_HOME)/Tools/SDFGen
#include $(SDF_LOC)/Make.package
#INCLUDE_LOCATIONS += $(SDF_LOC)
#VPATH_LOCATIONS += $(SDF_LOC)

vpath %.c   : $(VPATH_LOCATIONS)
vpath %.h   : $(VPATH_LOCATIONS)
vpath %.cpp : $(VPATH_LOCATIONS)
vpath %.H   : $(VPATH_LOCATIONS)
vpath %.F   : $(VPATH_LOCATIONS)
vpath %.f   : $(VPATH_LOCATIONS)
vpath %.f90 : $(VPATH_LOCATIONS)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
