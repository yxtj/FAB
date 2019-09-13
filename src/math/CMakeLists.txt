set(HEADERS
	norm.h
	activation_func.h
	accumulate.h
)
set(SOURCES
	norm.cpp
	activation_func.cpp
	accumulate.cpp
)
add_library(math
	${HEADERS} ${SOURCES})
