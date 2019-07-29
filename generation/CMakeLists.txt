

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)


project( generation )

set( CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} )

find_package( Geant4 REQUIRED ) 
include(${Geant4_USE_FILE})


include_directories( SYSTEM ${Geant4_INCLUDE_DIRS} )
include_directories( include )
aux_source_directory( src generation_src )

add_executable( generation ${generation_src}  generate.cc )
target_link_libraries( generation ${Geant4_LIBRARIES} )

install( TARGETS generation RUNTIME DESTINATION bin )
