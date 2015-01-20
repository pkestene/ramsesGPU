/**
 * \file DemoWindow.h
 * \brief Original example use of class GlutWindow.
 *
 * $Id: DemoWindow.h 1784 2012-02-21 10:34:58Z pkestene $
 */
////////////////////////////////////////////////////////////////
//                                                            //
// demoWindow.h                                               //
// beta version 0.3 - 9/9/97)                                 //
//                                                            //
// George Stetten and Korin Crawford                          //
// copyright given to the public domain                       //
//                                                            //
// Please email comments to email@stetten.com                 //
//                                                            //
////////////////////////////////////////////////////////////////


#ifndef DEMO_WINDOW_H_
#define DEMO_WINDOW_H_

#include "GlutMaster.h"

class DemoWindow : public GlutWindow{
public:

   int          height, width;
   int          initPositionX, initPositionY;

   DemoWindow(GlutMaster * glutMaster,
              int setWidth, int setHeight,
              int setInitPositionX, int setInitPositionY,
              const char* title);
   ~DemoWindow();

   void CallBackDisplayFunc(void);
   void CallBackReshapeFunc(int w, int h);   
   void CallBackIdleFunc(void);

   void StartSpinning(GlutMaster * glutMaster);
};

#endif // DEMO_WINDOW_H_
