#include "logging.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

void initLogger(int argc, char* argv[]){
	START_EASYLOGGINGPP(argc, argv);
	
	el::Configurations defaultConf;
	defaultConf.setToDefault();
	// Values are always std::string
	defaultConf.setGlobally(el::ConfigurationType::ToFile, "false");
	#ifndef NDEBUG
	defaultConf.setGlobally(el::ConfigurationType::Format,
		"%datetime{%H:%m:%s.%g} (%thread) %level %fbase:%line] %msg");
	#else
	defaultConf.setGlobally(el::ConfigurationType::Format,
		"%datetime{%H:%m:%s.%g} (%thread) %level] %msg");
	#endif
	defaultConf.set(el::Level::Debug, 
		el::ConfigurationType::Format, "%datetime{%H:%m:%s.%g} (%thread) %level %fbase:%line] %msg");

	// default logger uses default configurations
	el::Loggers::reconfigureLogger("default", defaultConf);
	el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
}

void setLogThreadName(const std::string& name){
	el::Helpers::setThreadName(name);
}
