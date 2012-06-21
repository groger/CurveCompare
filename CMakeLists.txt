cmake_minimum_required(VERSION 2.6)
 
PROJECT(CurveCompare)
#PROJECT(testAffichage)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(${PROJECT_SOURCE_DIR})

FIND_PACKAGE(GenerateCLP REQUIRED)
IF(GenerateCLP_FOUND)
  INCLUDE(${GenerateCLP_USE_FILE})
ELSE(GenerateCLP_FOUND)
  MESSAGE(FATAL_ERROR
          "GenerateCLP not found. Please set GenerateCLP_DIR.")
ENDIF(GenerateCLP_FOUND)

SET(src CurveCompare.cxx)
GENERATECLP(src CurveCompare.xml)
ADD_EXECUTABLE( CurveCompare ${src})

TARGET_LINK_LIBRARIES( CurveCompare ${ITK_LIBRARIES} vtkIO vtkHybrid vtkCommon ${VTK_LIBRARIES})
