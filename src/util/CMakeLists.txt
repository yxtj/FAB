set(HEADERS
	Timer.h
	#FileEnumerator.h
	Util.h
)
set(SOURCES
	Timer.cpp
	#FileEnumerator.cpp
	Util.cpp
)
add_library(util
	${HEADERS} ${SOURCES})