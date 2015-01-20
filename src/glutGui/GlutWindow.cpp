/**
 * \file GlutWindow.cpp
 * \brief Implements a base class for handling Glut interface callbacks.
 * 
 * \see http://www.stetten.com/george/glutmaster/glutmaster.html
 *
 * $Id: GlutWindow.cpp 1784 2012-02-21 10:34:58Z pkestene $
 */

////////////////////////////////////////////////////////////////
//                                                            //
// glutWindow.c++                                             //
// beta version 0.3 - 9/9/97)                                 //
//                                                            //
// George Stetten and Korin Crawford                          //
// copyright given to the public domain                       //
//                                                            //
// Please email comments to email@stetten.com                 //
//                                                            //
////////////////////////////////////////////////////////////////

#include "GlutWindow.h"

GlutWindow::GlutWindow(void){

}

GlutWindow::~GlutWindow(){

}

void GlutWindow::CallBackDisplayFunc(void){

                             //dummy function
}

void GlutWindow::CallBackIdleFunc(void){

                             //dummy function
}

void GlutWindow::CallBackKeyboardFunc(unsigned char key, int x, int y){

  (void) key; (void) x; (void) y;                //dummy function
}

void GlutWindow::CallBackMotionFunc(int x, int y){

   (void) x; (void) y;                     //dummy function
}

void GlutWindow::CallBackMouseFunc(int button, int state, int x, int y){

   (void) button; (void) state; (void) x; (void) y;      //dummy function
}

void GlutWindow::CallBackPassiveMotionFunc(int x, int y){

   (void) x; (void) y;                     //dummy function
}

void  GlutWindow::CallBackReshapeFunc(int w, int h){

   (void) w; (void) h;                     //dummy function
}
   
void GlutWindow::CallBackSpecialFunc(int key, int x, int y){

   (void) key; (void) x; (void) y;
}   

void GlutWindow::CallBackVisibilityFunc(int visible){

   (void) visible;                  //dummy function
}

void GlutWindow::SetWindowID(int newWindowID){

   windowID = newWindowID;
}

int GlutWindow::GetWindowID(void){

   return( windowID );
}

