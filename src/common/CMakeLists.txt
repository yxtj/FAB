set(HEADERS
	Option.h
	Statistics.h
)
set(SOURCES
	Option.cpp
)
add_library(common
	${HEADERS} ${SOURCES})
#target_link_libraries(common util)