set(HEADERS
	DataPoint.h
	DataHolder.h
	Loader.h
)
set(SOURCES
	DataHolder.cpp
	Loader.cpp
)
add_library(data
	${HEADERS} ${SOURCES})