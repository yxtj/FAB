set(HEADERS
	mathfunc.h
	Kernel.h
	KernelFactory.h
	Parameter.h
	ParameterProxy.h
	Proxy.h
	Model.h
	LogisticRegression.h
	MLP.h
	CNN.h
)
set(SOURCES
	mathfunc.cpp
	Kernel.cpp
	KernelFactory.cpp
	Parameter.cpp
	ParameterProxy.cpp
	Proxy.cpp
	Model.cpp
	LogisticRegression.cpp
	MLP.cpp
	CNN.cpp
)
add_library(model
	${HEADERS} ${SOURCES})
target_link_libraries(model data util)
