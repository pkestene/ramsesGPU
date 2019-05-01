/**
 * \file ConfigMap.h
 * \brief Define an object will allow to easily retrieve parameter from a dictionary.
 *
 * \date 12 November 2010
 * \author Pierre Kestener.
 *
 * $Id: ConfigMap.h 1783 2012-02-21 10:20:07Z pkestene $
 */
#ifndef CONFIG_MAP_H_
#define CONFIG_MAP_H_

#include "inih/INIReader.h"

/**
 * \class ConfigMap ConfigMap.h
 * \brief This is a specialized version of INIReader which reads and parses a INI
 * file into a key-value map (implemented using std::map). This class
 * is usefull to gather parameters.
 */
class ConfigMap : public INIReader
{
public:
  ConfigMap(std::string filename);
  ConfigMap(char* &buffer, int buffer_size);
  ~ConfigMap();

  //! Get a floating point value from the map.
  float getFloat(std::string section, std::string name, float default_value) const;
  
  //! Set a floating point value to a section/name.
  void setFloat(std::string section, std::string name, float value);
  
  //! Get a boolean value from the map.
  bool  getBool (std::string section, std::string name, bool default_value) const;

  //! Set a boolean value to a section/name.
  void  setBool (std::string section, std::string name, bool value);

}; // class ConfigMap

/**
 * Builds a ConfigMap object from the input parameter file.
 *
 * ConfigMap is return by value here.
 */
ConfigMap broadcast_parameters(std::string filename);


#endif // CONFIG_MAP_H_
