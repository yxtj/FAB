#pragma once

typedef int msg_t;

// Channel Type
struct CType{
	static constexpr int NormalControl = 0;
	static constexpr int ImmediateControl = 1;
	static constexpr int Data = 2;
};

// Message Type
struct MType {
	// Basic Control Signal (0-19)
	static constexpr int CReply = 0;
	static constexpr int COnline = 1;
	static constexpr int CWorkers = 2;
	static constexpr int CDataset = 3;
	static constexpr int CReady = 4;
	static constexpr int CStart = 5;

	// Working Control (0-19)
	static constexpr int CTrainPause = 10;
	static constexpr int CTrainContinue = 11;

	// Immediate Control (20-29)
	static constexpr int CTerminate = 20;
	static constexpr int CClosed = 21;

	// Data and data Request (30-39)
	static constexpr int DParameter = 30; // data for parametr
	static constexpr int DRParameter = 31; // data request for parameter
	static constexpr int DDelta = 34;
	static constexpr int DRDelta = 35;
	static constexpr int DReport= 36;
	static constexpr int DRReport= 37;

	// Process and Progress for Termination (40-49)
	static constexpr int PApply = 40;
	static constexpr int PSend = 41;
	static constexpr int PReport = 42;
	static constexpr int PRequest = 43;
	static constexpr int PFinish = 44;

	// Staticstics (60-69)
	static constexpr int SGather = 60;
};