/**
 * \file DemoWindow.cpp
 * \brief Original example use of class GlutWindow.
 *
 * $Id: DemoWindow.cpp 1784 2012-02-21 10:34:58Z pkestene $
 */
////////////////////////////////////////////////////////////////
//                                                            //
// demoWindow.c++                                             //
// beta version 0.3 - 9/9/97)                                 //
//                                                            //
// George Stetten and Korin Crawford                          //
// copyright given to the public domain                       //
//                                                            //
// Please email comments to email@stetten.com                 //
//                                                            //
////////////////////////////////////////////////////////////////


#include "DemoWindow.h"

DemoWindow::DemoWindow(GlutMaster * glutMaster,
                       int setWidth, int setHeight,
                       int setInitPositionX, int setInitPositionY,
                       const char* title){

   width  = setWidth;               
   height = setHeight;

   initPositionX = setInitPositionX;
   initPositionY = setInitPositionY;

   glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
   glutInitWindowSize(width, height);
   glutInitWindowPosition(initPositionX, initPositionY);
   glViewport(0, 0, width, height);   // This may have to be moved to after the next line on some platforms

   glutMaster->CallGlutCreateWindow(title, this);

   glEnable(GL_DEPTH_TEST);

   glMatrixMode(GL_PROJECTION);
   glOrtho(-80.0, 80.0, -80.0, 80.0, -500.0, 500.0);

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glRotatef(60, 1, 1, 1);
   glColor4f(1.0, 0.0, 0.0, 1.0);
}

DemoWindow::~DemoWindow(){

   glutDestroyWindow(windowID);
}

void DemoWindow::CallBackDisplayFunc(void){

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glColor4f(1.0, 0.0, 0.0, 1.0);
   glutWireSphere(50, 10, 10);

   glutSwapBuffers();
}

void DemoWindow::CallBackReshapeFunc(int w, int h){

   width = w;
   height= h;

   glViewport(0, 0, width, height); 
   CallBackDisplayFunc();
}

void DemoWindow::CallBackIdleFunc(void){

   glRotatef(0.25, 1, 1, 2);
   CallBackDisplayFunc();
}

void DemoWindow::StartSpinning(GlutMaster * glutMaster){

   glutMaster->SetIdleToCurrentWindow();
   glutMaster->EnableIdleFunction();
}
   






