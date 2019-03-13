set(HEADERS
	DataPoint.h
	DataHolder.h
)
set(SOURCES
	DataPoint.cpp
	DataHolder.cpp
)
add_library(data
	${HEADERS} ${SOURCES})