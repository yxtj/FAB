#pragma once

typedef int msg_t;

struct MType {
	// Basic Control
	static constexpr int CReply = 0;
	static constexpr int COnline = 1;
	static constexpr int CRegister = 2;
	static constexpr int CWorkers = 3;
	static constexpr int CDataset = 4;
	static constexpr int CTerminate = 7;
	static constexpr int CClosed = 8;
	static constexpr int CAlive = 9;

	// Data and data Request
	static constexpr int DParameter = 10;
	static constexpr int DRParameter = 11;
	static constexpr int DDelta = 15;
	static constexpr int DRDelta = 16;

	// Working Control
	static constexpr int CTrainPause = 20;
	static constexpr int CTrainContinue = 21;

	// Process and Progress (Termination)
	static constexpr int PApply = 40;
	static constexpr int PSend = 41;
	static constexpr int PReport = 42;
	static constexpr int PRequest = 43;
	static constexpr int PFinish = 44;

	// Staticstics
	static constexpr int SGather = 60;
};