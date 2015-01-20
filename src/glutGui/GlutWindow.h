/**
 * \file GlutWindow.h
 * \brief Defines a base class for handling Glut interface callbacks.
 * 
 * \see http://www.stetten.com/george/glutmaster/glutmaster.html
 *
 * $Id: GlutWindow.h 1784 2012-02-21 10:34:58Z pkestene $ 
 */

////////////////////////////////////////////////////////////////
//                                                            //
// glutWindow.h                                               //
// beta version 0.3 - 9/9/97)                                 //
//                                                            //
// George Stetten and Korin Crawford                          //
// copyright given to the public domain                       //
//                                                            //
// Please email comments to email@stetten.com                 //
//                                                            //
////////////////////////////////////////////////////////////////


#ifndef GLUT_WINDOW_H_
#define GLUT_WINDOW_H_

/**
 * \class GlutWindow GlutWindow.h
 * \brief Handle Glut interface callbacks.
 */
class GlutWindow {
protected:

   int          windowID;

public:

   GlutWindow(void);
   ~GlutWindow();

   virtual void CallBackDisplayFunc();
   virtual void CallBackIdleFunc(void);
   virtual void CallBackKeyboardFunc(unsigned char key, int x, int y);
   virtual void CallBackMotionFunc(int x, int y);
   virtual void CallBackMouseFunc(int button, int state, int x, int y);
   virtual void CallBackPassiveMotionFunc(int x, int y);
   virtual void CallBackReshapeFunc(int w, int h);   
   virtual void CallBackSpecialFunc(int key, int x, int y);   
   virtual void CallBackVisibilityFunc(int visible);

   void    SetWindowID(int newWindowID);
   int     GetWindowID(void);

};

#endif // GLUT_WINDOW_H_



