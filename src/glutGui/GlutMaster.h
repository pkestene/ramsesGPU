/**
 * \file GlutMaster.h
 * \brief Defines a class for handling multiple GlutWindows.
 * 
 * \see http://www.stetten.com/george/glutmaster/glutmaster.html
 *
 * $Id: GlutMaster.h 1784 2012-02-21 10:34:58Z pkestene $
 */

////////////////////////////////////////////////////////////////
//                                                            //
// glutMaster.h                                               //
// beta version 0.3 - 9/9/97)                                 //
//                                                            //
// George Stetten and Korin Crawford                          //
// copyright given to the public domain                       //
//                                                            //
// Please email comments to email@stetten.com                 //
//                                                            //
////////////////////////////////////////////////////////////////

#ifndef GLUT_MASTER_H_
#define GLUT_MASTER_H_

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "GlutWindow.h"

#define MAX_NUMBER_OF_WINDOWS 256 

/**
 * \class GlutMaster GlutMaster.h
 * \brief A C++ wrapper class around Glut resources.
 */
class GlutMaster {

private:

   static void CallBackDisplayFunc(void);
   static void CallBackIdleFunc(void); 
   static void CallBackKeyboardFunc(unsigned char key, int x, int y);
   static void CallBackMotionFunc(int x, int y);
   static void CallBackMouseFunc(int button, int state, int x, int y);
   static void CallBackPassiveMotionFunc(int x, int y);
   static void CallBackReshapeFunc(int w, int h); 
   static void CallBackSpecialFunc(int key, int x, int y);   
   static void CallBackVisibilityFunc(int visible);

   static int currentIdleWindow;
   static int idleFunctionEnabled;

public:
 
   GlutMaster();
   ~GlutMaster();
    
   void  CallGlutCreateWindow(const char* setTitle, GlutWindow * glutWindow);
   void  CallGlutMainLoop(void);

   void  DisableIdleFunction(void);
   void  EnableIdleFunction(void);
   int   IdleFunctionEnabled(void);

   int   IdleSetToCurrentWindow(void);
   void  SetIdleToCurrentWindow(void);
};

#endif // GLUT_MASTER_H_



