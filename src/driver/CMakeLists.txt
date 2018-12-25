set(TOOLS
	tools/ReplyHandler.h
	tools/ReplyHandler.cpp
	tools/SyncUnit.h
	tools/SyncUnit.cpp
)

add_library(driver
	Dispatcher.hpp
	MsgDriver.h
	MsgDriver.cpp
	${TOOLS}
)