set(HEADERS
	Timer.h
	#FileEnumerator.h
	Sleeper.h
	Util.h
)
set(SOURCES
	Timer.cpp
	#FileEnumerator.cpp
	Sleeper.cpp
	Util.cpp
)
add_library(util
	${HEADERS} ${SOURCES})
target_link_libraries(util
	${CMAKE_THREAD_LIBS_INIT}
)
