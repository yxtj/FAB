set(HEADERS
	Trainer.h
	GD.h
	EM.h
	TrainerFactory.h
)
set(SOURCES
	Trainer.cpp
	GD.cpp
	EM.cpp
	TrainerFactory.cpp
)
add_library(train
	${HEADERS} ${SOURCES})