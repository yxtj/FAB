set(HEADERS
	type_traits_dummy.h
	is_container.h
	is_stl_container.h
	is_sequential_container.h
	is_associated_container.h
	is_iterator.h
	is_pair.h
	is_tuple.h
)
set(SOURCES
)
add_library(type_traits INTERFACE)
	#${HEADERS} ${SOURCES})
target_include_directories(type_traits INTERFACE ./)
