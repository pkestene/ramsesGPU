/**
 * \file ConfigMap.cpp
 * \brief Implement ConfigMap, essentially a INIReader with additional get methods.
 *
 * \date 12 November 2010
 * \author Pierre Kestener.
 *
 * $Id: ConfigMap.cpp 1783 2012-02-21 10:20:07Z pkestene $
 */
#include "ConfigMap.h"
#include <cstdlib> // for strtof
#include <sstream>


// =======================================================
// =======================================================
ConfigMap::ConfigMap(std::string filename) :
  INIReader(filename)
{
} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
ConfigMap::~ConfigMap()
{
} // ConfigMap::~ConfigMap

// =======================================================
// =======================================================
float ConfigMap::getFloat(std::string section, std::string name, float default_value)
{
  std::string valstr = getString(section, name, "");
  const char* value = valstr.c_str();
  char* end;
  // This parses "1234" (decimal) and also "0x4D2" (hex)
  float valFloat = strtof(value, &end);
  return end > value ? valFloat : default_value;
} // ConfigMap::getFloat

// =======================================================
// =======================================================
void ConfigMap::setFloat(std::string section, std::string name, float value)
{

  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setFloat

// =======================================================
// =======================================================
bool ConfigMap::getBool(std::string section, std::string name, bool default_value)
{
  bool val = default_value;
  std::string valstr = getString(section, name, "");
  
  if (!valstr.compare("1") or 
      !valstr.compare("yes") or 
      !valstr.compare("true") or
      !valstr.compare("on"))
    val = true;
  if (!valstr.compare("0") or 
      !valstr.compare("no") or 
      !valstr.compare("false") or
      !valstr.compare("off"))
    val=false;
  
  // if valstr is empty, return the default value
  if (!valstr.size())
    val = default_value;

  return val;

} // ConfigMap::getBool

// =======================================================
// =======================================================
void ConfigMap::setBool(std::string section, std::string name, bool value)
{

  if (value)
    setString(section, name, "true");
  else
    setString(section, name, "false");

} // ConfigMap::setBool

