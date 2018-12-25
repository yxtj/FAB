set(HEADERS
	easylogging++.h
	logging.h
)
set(SOURCES
	easylogging++.cc
	logging.cpp
)
SET(LOGGING_DEFINE "-DELPP_THREAD_SAFE -DELPP_STL_LOGGING -DELPP_LOG_UNORDERED_MAP" CACHE STRING
    "local defines for logging")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LOGGING_DEFINE}")
add_library(logging
	${HEADERS} ${SOURCES})
#target_compile_options(logging PRIVATE "${LOGGING_DEFINE}")
target_link_libraries(logging ${CMAKE_THREAD_LISBS_INIT})
