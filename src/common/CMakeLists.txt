set(HEADERS
	ConfData.h
	Statistics.h
)
set(SOURCES
	Statistics.cpp
)
add_library(common
	${HEADERS} ${SOURCES})