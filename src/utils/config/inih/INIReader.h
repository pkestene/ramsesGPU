/**
 * \file INIReader.h
 * \brief Original INI file parser from inih slightly modified.
 *
 * This has been modified to add a virtual destructor, make Get and 
 * GetInteger virtual, and format comments to Doxygen syntax. 
 */

// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info:
//
// http://code.google.com/p/inih/

#ifndef __INIREADER_H__
#define __INIREADER_H__

#include <map>
#include <string>

/**
 * \class INIReader INIReader.h
 * \brief Read an INI file into easy-to-access name/value pairs. (Note that I've gone
 * for simplicity here rather than speed, but it should be pretty decent.)
 */
class INIReader
{
public:
  /**
   * Construct INIReader and parse given filename. See ini.h for more info
   * about the parsing.
   */
  INIReader(std::string filename);
  INIReader(char* &buffer, int buffer_size);
  virtual ~INIReader();

  /**
   * Return the result of ini_parse(), i.e., 0 on success, line number of
   * first error on parse error, or -1 on file open error.
   */
  int ParseError();
  
  /**
   * Get a string value from INI file, returning default_value if not found.
   */
  virtual std::string getString(std::string section, std::string name,
				std::string default_value) const;

  /**
   * Set a string value to section/name.
   */
  virtual void        setString(std::string section, std::string name,
				std::string value);
  
  /** Get an integer (long) value from INI file, returning default_value if
   * not found.
   */
  virtual long getInteger(std::string section, std::string name, long default_value) const;

  /** 
   * Set an integer value to an section/name.
   * reverse operation of set getInteger.
   */
  virtual void setInteger(std::string section, std::string name, long value);
  
  /**
   * \brief Print the content of a IniReader object
   */
  friend std::ostream& operator<<(std::ostream& os, const INIReader& cfg);
  
private:
  int _error;
  std::map<std::string, std::string> _values;
  static std::string makeKey(std::string section, std::string name);
  static int valueHandler(void* user, const char* section, const char* name,
			  const char* value);
};

#endif  // __INIREADER_H__
