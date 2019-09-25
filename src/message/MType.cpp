#include "MType.h"

constexpr int MType::CReply;
constexpr int MType::COnline;
constexpr int MType::CWorkers;
constexpr int MType::CDataset;
constexpr int MType::CReady;
constexpr int MType::CStart;

// Working Control (0-19)
constexpr int MType::CTrainPause;
constexpr int MType::CTrainContinue;
constexpr int MType::CReset;
constexpr int MType::CProbeDone;

// Immediate Control (20-29)
constexpr int MType::CTerminate;
constexpr int MType::CClosed;

// Data and data Request (30-39)
constexpr int MType::DParameter; // data for parametr
constexpr int MType::DRParameter; // data request for parameter
constexpr int MType::DDelta;
constexpr int MType::DRDelta;
constexpr int MType::DReport;
constexpr int MType::DRReport;
constexpr int MType::DLoss;
constexpr int MType::DRLoss;

// Process and Progress for Termination (40-49)
constexpr int MType::PApply;
constexpr int MType::PSend;
constexpr int MType::PReport;
constexpr int MType::PRequest;
constexpr int MType::PFinish;

// Configuration and Meta data (50-59)
constexpr int MType::FSizeConf;
constexpr int MType::FGlobalBatchSize;
constexpr int MType::FLocalReportSize;

// Staticstics (60-69)
constexpr int MType::SGather;
