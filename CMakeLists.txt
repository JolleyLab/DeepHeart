cmake_minimum_required(VERSION 2.8.9)

project(DeepHeart)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_CATEGORY "Cardiac")
set(EXTENSION_CONTRIBUTORS "Christian Herz (CHOP), Andras Lasso (PerkLab, Queen's), Matthew Jolley (CHOP)")
set(EXTENSION_DESCRIPTION "Modules for the segmentation of heart valves using DeepLearning")
set(EXTENSION_DEPENDS SlicerHeart)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(slicer/DeepHeart)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
