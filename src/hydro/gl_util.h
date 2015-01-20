/**
 * \file gl_util.h
 * \brief Defines a macro handling OpenGL errors.
 *
 *
 * $Id: gl_util.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef _GL_UTIL_H_
#define _GL_UTIL_H_

// the following macro is copied from Inria/axel; 
// see file raycaster/axl/QCUDAImplicitRayCaster/include/QCUDAImplicitRayCaster/QCIRC_internal.hpp

// Beware that it is not legal to gl glGetError inside a glBegin/glEnd-block!
#ifndef GL_SAFE_CALL
#define GL_SAFE_CALL( call ) do {		\
    call;					\
    GLenum err = glGetError();						\
    if ( err != GL_NO_ERROR ) {						\
      fprintf(stderr, "OpenGL error in file '%s' in line %i : %s.\n",	\
	      __FILE__, __LINE__, gluErrorString( err ) );              \
      exit(EXIT_FAILURE);						\
    } } while (0)
#endif // GL_SAFE_CALL

#endif // _GL_UTIL_H_
