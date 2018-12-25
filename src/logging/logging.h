#pragma once
/**
 * This file is used to simplify the usage of logging modules.
 * If some other logger module is used, just change the include line.
 */

#include "easylogging++.h"
#include <string>

void initLogger(int argc, char* argv[]);
void setLogThreadName(const std::string& name);
