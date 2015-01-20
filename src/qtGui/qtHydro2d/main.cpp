/**
 * \file qtHydro2d/main.cpp
 * \brief QT-based applications to display hydro simulations results
 * in real-time.
 *
 * \note Note that the GPU-based computations uses the CUDA/OpenGL
 * inter-operability routines.
 *
 * \date 11-03-2010
 * \author Pierre Kestener
 */

#include <QApplication>

#include "MainWindow.h"

#include <HydroRunGodunov.h>
#include <HydroRunKT.h>

using hydroSimu::HydroRunGodunov;
using hydroSimu::HydroRunKT;
using hydroSimu::HydroRunBase;

#ifdef __CUDACC__
#include "HydroWidgetGpu.h"
#else
#include "HydroWidget.h"
#endif // __CUDACC__

/** Parse configuration parameter file */
#include "GetPot.h"
#include <ConfigMap.h>

/* Graphics Magick C++ API to dump PNG image files */
#ifdef USE_GM
#include <Magick++.h>
#endif // USE_GM

void print_help(int argc, char* argv[]);
void make_default_paramFile(std::string fileName);

/*******************************
 *******************************/
int main(int argc, char* argv[])
{

  /* 
   * read parameters from parameter file or command line arguments
   */
  
  /* parse command line arguments */
  GetPot cl(argc, argv);
  
  /* search for multiple options with the same meaning HELP */
  if( cl.search(3, "--help", "-h", "--sos") ) {
    print_help(argc,argv);
    exit(0);
  }

  /* set default configuration parameter fileName */
  const std::string default_input_file = std::string(argv[0])+ ".ini";

  /* 
   * try to find if a parameter filename was given on the command line
   * if not use a built-in one
   */
  if( !cl.search(1, "--param") ) {
    std::cout << "[Warning] You did not provide a .ini parameter file !\n";
    std::cout << "[Warning] Using the built-in one\n";
    std::cout << "[Warning] Make test parameter file : " << default_input_file << std::endl;
    make_default_paramFile(default_input_file);
  } 
  
  // set parameter filename from argument to option --param
  const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");
  
  // set numerical scheme from command line argument
  const std::string numSchemeStr  = cl.follow("godunov", "--scheme");
  bool useGodunov     = !numSchemeStr.compare("godunov");
  bool useKurganov    = !numSchemeStr.compare("kurganov");
  bool useRelaxingTVD = !numSchemeStr.compare("relaxingTVD");

  NumScheme numScheme = GODUNOV;
  if (useKurganov)
    numScheme = KURGANOV;
  if (useRelaxingTVD)
    numScheme = RELAXING;

  /* parse parameters from input file */
  ConfigMap configMap(input_file); 

  if (useRelaxingTVD) // make sure we use 3 ghost cells on borders
    configMap.setInteger("mesh","ghostWidth", 3);

  /* dump input file and exit */
  if( cl.search(2, "--dump-param-file", "-d") ) {
    std::cout << configMap << std::endl;
    exit(0);
  }

  /* set traceVersion and doTrace :
   * doTrace parameter (trigger trace characteristic computations) */
  int traceVersion = configMap.getInteger("hydro", "traceVersion", 1);
  bool traceEnabled = (traceVersion == 0) ? false : true;

  /* QT Window title string */
  std::string winTitle;
#ifdef __CUDACC__
  if (useGodunov)
    winTitle = std::string("2D Euler simulation: Godunov scheme -- GPU");
  else if (useKurganov)
    winTitle = std::string("2D Euler simulation: Kurganov scheme -- GPU");
#else // CPU version
  if (useGodunov)
    winTitle = std::string("2D Euler simulation: Godunov scheme -- CPU");
  else if (useKurganov)
    winTitle = std::string("2D Euler simulation: Kurganov scheme -- CPU");
#endif // __CUDACC__

  std::cout << " QT keyboard shortcuts:                      " << std::endl;
  std::cout << " r      : reset simulation; same initialization" << std::endl;
  std::cout << " R      : reset simulation; new  initialization" << std::endl;
  std::cout << " c      : switch color/grey map              " << std::endl;
  std::cout << " [SPACE]: start/stop simulation              " << std::endl;
  std::cout << " s      : start/stop simulation              " << std::endl;
  std::cout << " a      : increase maxvar (colormap)         " << std::endl;
  std::cout << " A      : decrease maxvar (colormap)         " << std::endl;
  std::cout << " b      : increase minvar (colormap)         " << std::endl;
  std::cout << " B      : decrease minvar (colormap)         " << std::endl;
  std::cout << " u      : change plot variable (rho, u, v, E)" << std::endl;
  std::cout << " [ESC]  : quit                               " << std::endl;
  std::cout << " q      : quit                               " << std::endl;
  
  int posX = cl.follow(200,    "--posx");
  int posY = cl.follow(100,    "--posy");

  // when using PNG output, we must initialize GraphicsMagick library
#ifdef USE_GM
  Magick::InitializeMagick("");
#endif

  QApplication app(argc, argv);

  // create here hydroRun object instead of inside HydroWidget (which
  // was a conception error after all)
  HydroRunBase* hydroRun;
  if (useGodunov) {
    hydroRun = (HydroRunBase *) new HydroRunGodunov(configMap);

    HydroRunGodunov *hydroRunGodunov = dynamic_cast<HydroRunGodunov*>(hydroRun);
    hydroRunGodunov->setTraceEnabled(traceEnabled);
    hydroRunGodunov->setTraceVersion(traceVersion);
  } else if (useKurganov) {
    hydroRun = (HydroRunBase *) new HydroRunKT(configMap);
  }

#ifdef __CUDACC__
  HydroWidget* hydroWidget = (HydroWidget *) new HydroWidgetGpu(configMap, hydroRun, 0);
#else
  HydroWidget* hydroWidget = (HydroWidget *) new HydroWidget(configMap, hydroRun, 0);
#endif // __CUDACC__

  MainWindow mainWindow(configMap, numScheme, hydroWidget, 0);
  QRect geom =  mainWindow.geometry();
  mainWindow.move(posX,posY);
  mainWindow.show();
  mainWindow.hydroWidget->startSimulation();
  return app.exec();
}

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  (void) argc;
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << argv[0] << " is a C++ version of hydro2d with a QT gui: " << endl;
  cout << "solve 2D Euler equations on a cartesian grid." << endl;
  cout << endl; 
  cout << "USAGE:" << endl;
  cout << "--help, -h, --sos" << endl;
  cout << "        get some help about this program." << endl << endl;
  cout << "--param [string]" << endl;
  cout << "        specify configuration parameter file (INI format)" << endl;
  cout << "--scheme [string]" << endl;
  cout << "        string parameter must be godunov or kurganov (numerical scheme)" << endl;
  cout << endl << endl;       
  cout << "--dump-param-file, -d" << endl;
  cout << "        show contents of database that was created by file parser.";
  cout << endl;
}

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void make_default_paramFile(std::string fileName)
{
  std::fstream outFile;
  outFile.open (fileName.c_str(), std::ios_base::out);

  outFile << "# default parameter file for hydrodynamics simulations\n";
  outFile << "# Generated on " << __DATE__ << " " << __TIME__ << std::endl;
  outFile <<                                std::endl;

  outFile << "[run]"                     << std::endl;
  outFile << "tend=0.2"                  << std::endl;
  outFile << "noutput=100"               << std::endl;
  outFile << "nstepmax=2000"             << std::endl;
  outFile <<                                std::endl;

  outFile << "[mesh]"                    << std::endl;
  outFile << "nx=150"                    << std::endl;
  outFile << "ny=600"                    << std::endl;
  outFile << "nz=1"                      << std::endl;
  outFile << "# BoundaryConditionType :" << std::endl;
  outFile << "# BC_UNDEFINED=0"          << std::endl;
  outFile << "# BC_DIRICHLET=1"          << std::endl;
  outFile << "# BC_NEUMANN=2"            << std::endl;
  outFile << "# BC_PERIODIC=3"           << std::endl;
  outFile << "boundary_xmin=1"           << std::endl;
  outFile << "boundary_xmax=1"           << std::endl;
  outFile << "boundary_ymin=1"           << std::endl;
  outFile << "boundary_ymax=1"           << std::endl;
  outFile << "boundary_zmin=1"           << std::endl;
  outFile << "boundary_zmax=1"           << std::endl;
  outFile <<                                std::endl;

  outFile << "[hydro]"                   << std::endl;
  outFile << "problem=jet"               << std::endl;
  outFile << "cfl=0.31"                  << std::endl;
  outFile << "niter_riemann=10"          << std::endl;
  outFile << "iorder=2"                  << std::endl;
  outFile << "slope_type=2.0"            << std::endl;
  outFile << "scheme=muscl"              << std::endl;
  outFile << "traceVersion=1"            << std::endl;
  outFile << "riemannSolver=approx"      << std::endl;
  outFile << "smallr=1e-9"               << std::endl;
  outFile << "smallc=1e-8"               << std::endl;
  outFile <<                                std::endl;
  
  outFile << "[jet]"                     << std::endl;
  outFile << "ijet=10"                   << std::endl;
  outFile << "djet=1."                   << std::endl;
  outFile << "ujet=300."                 << std::endl;
  outFile << "pjet=1."                   << std::endl;
  outFile << "offsetJet=30"              << std::endl;
  outFile <<                                std::endl;
 
  outFile << "[visu]"                    << std::endl;
  outFile << "minvar=0.0"                << std::endl;
  outFile << "maxvar=14.0"               << std::endl;
  outFile << "manualContrast=0"          << std::endl;
  outFile <<                                std::endl;

  outFile << "[output]"                  << std::endl;
  outFile << "latexAnimation=no"         << std::endl;
  outFile << "outputMode=binary"         << std::endl;
  outFile << "outputDir=./"              << std::endl;
  outFile << "outputPrefix=jet"          << std::endl;
  outFile << "outputHdf5=yes"            << std::endl;
  outFile << "outputVtk=no"              << std::endl;
  outFile << "colorPng=no"               << std::endl;
  outFile <<                                std::endl;

  outFile.close();
}
