set(HEADERS
	RPCInfo.h
	Task.h
	NetworkThread.h
	NetworkImplMPI.h
)
set(SOURCES
	Task.cpp
	NetworkThread.cpp
	NetworkImplMPI.cpp
)
add_library(network2
	${HEADERS} ${SOURCES})
target_include_directories(network2 PRIVATE ${MPI_CXX_INCLUDE_PATH})
set_target_properties(network2 PROPERTIES
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
target_link_libraries(network2 ${MPI_CXX_LIBRARIES})

