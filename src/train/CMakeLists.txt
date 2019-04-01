set(HEADERS
	Trainer.h
	GD.h
	EM.h
	EM_KMeans.h
	TrainerFactory.h
)
set(SOURCES
	Trainer.cpp
	GD.cpp
	EM.cpp
	EM_KMeans.cpp
	TrainerFactory.cpp
)
add_library(train
	${HEADERS} ${SOURCES})