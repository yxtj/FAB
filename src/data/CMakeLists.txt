set(HEADERS
	DataPoint.h
	DataHolder.h
	DataLoader.h
)
set(SOURCES
	DataPoint.cpp
	DataHolder.cpp
	DataLoader.cpp
)
add_library(data
	${HEADERS} ${SOURCES})