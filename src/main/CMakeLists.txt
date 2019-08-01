set(HEADERS
	Option.h
	CLI11.hpp
	func.h
	ParameterIO.h
)
set(SOURCES
	Option.cpp
	func.cpp
	ParameterIO.cpp
)
set(LOCAL_UTIL
	CLI11.hpp
	func.h
	ParameterIO.h
	func.cpp
	ParameterIO.cpp
)

# exe: main
add_executable(main main.cpp
	${HEADERS} ${SOURCES})
target_link_libraries(main
#	common data model train util
	distr #driver logging network 
	Boost::program_options
)

# exe: postprocess
add_executable(postprocess postprocess.cpp
	${LOCAL_UTIL})
target_link_libraries(postprocess
	data model util
)
# exe: generate
add_executable(generate generate.cpp
	${LOCAL_UTIL})
target_link_libraries(generate
	model util
)

# exe: calcgradient
add_executable(calcgradient calcgradient.cpp
	${LOCAL_UTIL})
target_link_libraries(calcgradient
	data model util
)
# exe: calcpriority
add_executable(calcpriority calcpriority.cpp
	${LOCAL_UTIL})
target_link_libraries(calcpriority
	data model util
)
