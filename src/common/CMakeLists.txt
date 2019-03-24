set(HEADERS
	Option.h
	Statistics.h
)
set(SOURCES
	Option.cpp
	Statistics.cpp
)
add_library(common
	${HEADERS} ${SOURCES})
target_link_libraries(common util Boost::program_options)