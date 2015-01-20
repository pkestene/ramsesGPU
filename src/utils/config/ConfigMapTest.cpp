/**
 * \file ConfigMapTest.cpp
 * \brief This is an example use of class ConfigMap.
 *
 * \date 15 Nov 2010
 * \author Pierre Kestener
 *
 * $Id: ConfigMapTest.cpp 1783 2012-02-21 10:20:07Z pkestene $
 */

#include <iostream>
#include <fstream>
#include "ConfigMap.h"

int main(int argc, char* argv[])
{
  // make test.ini file
  std::fstream iniFile;
  iniFile.open ("./test.ini", std::ios_base::out);
  iniFile << "; Test config file for ConfigMapTest.cpp" << std::endl;
  
  iniFile << "[Hydro]             ; Hydrodynamics simulations configuration" << std::endl;
  iniFile << "scheme=godunov      ; numerical scheme" << std::endl;
  
  iniFile << "[Jet]" << std::endl;
  iniFile << "# another type of comment" << std::endl;
  iniFile << "ijet = 10      ; injection width" << std::endl;
  iniFile << "djet = 3.5     ; jet density" << std::endl;
  iniFile << "[output]" << std::endl;
  iniFile << "useHdf5 = yes" << std::endl;
  iniFile << "outputDir=./"  << std::endl;
  iniFile << "out_type=png"  << std::endl;
  iniFile.close();

  // create a INIReader instance
  ConfigMap cfg("./test.ini");
  
  if (cfg.ParseError() < 0) {
    std::cout << "Can't load 'test.ini'\n";
    return 1;
  }
  std::cout << "Config loaded from 'test.ini':" << std::endl;
  std::cout << "Hydro::scheme : " << cfg.getString("Hydro", "scheme", "unknown") << std::endl;
  std::cout << "Jet::ijet : " << cfg.getInteger("Jet", "ijet", -1) << std::endl;
  std::cout << "modify Jet::ijet " << std::endl;
  cfg.setInteger("Jet", "ijet", 23);
  std::cout << "Jet::ijet : " << cfg.getInteger("Jet", "ijet", -1) << std::endl;
  std::cout << "Jet::djet : " << cfg.getFloat("Jet", "djet", -1.0) << std::endl;
  std::cout << "output::useHdf5 : " << cfg.getBool("output", "useHdf5", false) << std::endl;
  std::cout << "modify output::useHdf5 (if already existing)" << std::endl;
  cfg.setBool("output", "useHdf5", false);
  std::cout << "output::useHdf5 : " << cfg.getBool("output", "useHdf5", true) << std::endl;

  std::cout << "output::useVtk : " << cfg.getBool("output", "useVtk", true) << std::endl;
  std::cout << "modify output::useVtk (if already existing)" << std::endl;
  cfg.setBool("output", "useVtk", false);
  std::cout << "output::useVtk : " << cfg.getBool("output", "useVtk", true) << std::endl;
  std::cout << "DoNotExists::dummy : " << cfg.getFloat("DoNotExist", "dummy", -100.5) << std::endl;

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Print config :" << std::endl;
  std::cout << cfg;

  return 0;
}
