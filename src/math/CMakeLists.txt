set(HEADERS
	norm.h
	activation_func.h
	accumulate.h
	RandomGenerator.h
)
set(SOURCES
	norm.cpp
	activation_func.cpp
	accumulate.cpp
	RandomGenerator.cpp
)
add_library(math
	${HEADERS} ${SOURCES})
