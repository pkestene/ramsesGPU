/**
 * \file INIReader.cpp
 * \brief Implementation of class INIReader.
 *
 * Read an INI file into easy-to-access name/value pairs.
 */

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "ini.h"
#include "INIReader.h"

// =======================================================
// =======================================================
INIReader::INIReader(std::string filename)
{
  _error = ini_parse(filename.c_str(), valueHandler, this);
}

// =======================================================
// =======================================================
INIReader::INIReader(char* &buffer, int buffer_size)
{
  _error = ini_parse_buffer(buffer, buffer_size, valueHandler, this);
}

// =======================================================
// =======================================================
INIReader::~INIReader()
{
}

// =======================================================
// =======================================================
int INIReader::ParseError()
{
  return _error;
}

// =======================================================
// =======================================================
std::string INIReader::getString(std::string section, std::string name, std::string default_value) const
{
  const std::string key = makeKey(section, name);
  // for const correctness use method at instead of operator[]
  return _values.count(key) ? _values.at(key) : default_value;
}

// =======================================================
// =======================================================
void INIReader::setString(std::string section, std::string name, std::string value)
{
  std::string key = makeKey(section, name);
  _values[key] = value;
}

// =======================================================
// =======================================================
long INIReader::getInteger(std::string section, std::string name, long default_value) const
{
  std::string valstr = getString(section, name, "");
  const char* value = valstr.c_str();
  char* end;
  // This parses "1234" (decimal) and also "0x4D2" (hex)
  long n = strtol(value, &end, 0);
  return end > value ? n : default_value;
}

// =======================================================
// =======================================================
void INIReader::setInteger(std::string section, std::string name, long value)
{
  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());
}

// =======================================================
// =======================================================
std::ostream& operator<<(std::ostream &os, const INIReader& cfg)
{
  std::map<std::string,std::string>::const_iterator it;
  for (it=cfg._values.begin(); it != cfg._values.end(); it++){
    os << (*it).first << " = " << (*it).second << "\n";
  }
  return os;
}

// =======================================================
// =======================================================
std::string INIReader::makeKey(std::string section, std::string name)
{
  std::string key = section + "." + name;
  // Convert to lower case to make lookups case-insensitive
  for (unsigned int i = 0; i < key.length(); i++)
    key[i] = tolower(key[i]);
  return key;
}

// =======================================================
// =======================================================
int INIReader::valueHandler(void* user, const char* section, const char* name,
                            const char* value)
{
  INIReader* reader = (INIReader*)user;
  reader->_values[makeKey(section, name)] = value;
  return 1;
}
