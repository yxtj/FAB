set(HEADERS
	IDMapper.h
	Runner.h
	Master.h
	Worker.h
)
set(SOURCES
	IDMapper.cpp
	Runner.cpp
	Master.cpp
	Worker.cpp
)
add_library(distr
	${HEADERS} ${SOURCES})
target_link_libraries(distr
	common data model train util
	driver logging network 
)