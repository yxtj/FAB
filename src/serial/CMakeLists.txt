set(HEADERS
	serialization.h
	serialize_general.hpp
	serializer_basic.hpp
	serializer_iter.hpp
	serializer_tuple34.hpp
)
set(SOURCES
)
add_library(serial INTERFACE)
#	${HEADERS} ${SOURCES})
target_include_directories(serial INTERFACE ./)
