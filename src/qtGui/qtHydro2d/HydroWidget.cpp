/**
 * \file HydroWidget.cpp
 * \brief Implements class HydroWidget.
 *
 * \author Pierre Kestener
 * \date 14-03-2010
 */
#include <GL/glew.h>

#include <QtGui>
#include <QtOpenGL>
#include <QSizePolicy>

#include <math.h>
#include <cstdio>

// application header
#include "HydroWidget.h"

#include "gl_util.h"

// =======================================================
// =======================================================
HydroWidget::HydroWidget(ConfigMap& _param, HydroRunBase* _hydroRun, QWidget* parent)
  :   QGLWidget(parent), nx(0), ny(0),
      gl_PBO(0), gl_Tex(0), 
      param(_param), hydroRun(_hydroRun),
      t(0.0), dt(0.0), nStep(0), nbStepToDo(0), animate(true), 
      useColor(0), manualContrast(0),
      currentRiemannConfigNb(0),
      minvar(0.0), maxvar(0.0), displayVar(0)
      
{
  
  nx     = param.getInteger("mesh", "nx", 100);
  ny     = param.getInteger("mesh", "ny", 100);

  currentRiemannConfigNb = 0;

  std::cout << "Init Hydro2d run\n";
  std::string problem = param.getString("hydro", "problem", "unknown");
  if (!problem.compare("jet")) {
    hydroRun->init_hydro_jet();
  } else if (!problem.compare("implode")) {
    hydroRun->init_hydro_implode();
  } else if (!problem.compare("blast")) {
    hydroRun->init_hydro_blast();
  } else if (!problem.compare("Kelvin-Helmholtz")) {
    hydroRun->init_hydro_Kelvin_Helmholtz();
  } else if (!problem.compare("Rayleigh-Taylor")) {
    hydroRun->init_hydro_Rayleigh_Taylor();
  } else if (!problem.compare("riemann2d")) {
    // get Riemann config number (integer between 0 and 18)
    currentRiemannConfigNb = param.getInteger("hydro", "riemann_config_number", 0);
    hydroRun->setRiemannConfId(currentRiemannConfigNb);
    hydroRun->init_hydro_Riemann();
  }

   // if using the unsplit scheme, we also need to initialize h_U2 (CPU only)
  bool unsplitEnabled = param.getBool("hydro","unsplit", true);
  /*if (unsplitEnabled) {
    hydroRun->getData2().copyHard(hydroRun->getData());
    }*/
 

  // init color map
  std::cout << "init colormap" << std::endl;
  initColormap();
  plot_rgba = new unsigned int[(nx+4)*(ny+4)];
  manualContrast = param.getInteger("visu", "manualContrast", 0);
  displayVar = 0;
  computeMinMax(hydroRun->getDataHost().data(),hydroRun->getDataHost().section(), displayVar);
  printf("constructor : min %f max %f\n",minvar,maxvar);

    // read animation param from config
  animate = param.getBool("output", "animate", true);

  // set some widget properties
  this->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

} // HydroWidget::HydroWidget

// =======================================================
// =======================================================
HydroWidget::~HydroWidget()
{
  //makeCurrent();

  delete [] plot_rgba;

  deletePBO();
  deleteTexture();

} // HydroWidget::~HydroWidget

// =======================================================
// =======================================================
QSize HydroWidget::minimumSizeHint() const
{
  return QSize(50, 50);
} // HydroWidget::minimumSizeHint


// =======================================================
// =======================================================
QSize HydroWidget::sizeHint() const
{
  return QSize(nx, ny);
} // HydroWidget::sizeHint

// =======================================================
// =======================================================
void HydroWidget::startSimulation() 
{

  startTimer(0);

} // HydroWidget::startSimulation

// =======================================================
// =======================================================
void HydroWidget::updateData() {

  // do one step of numerical scheme:
  if (animate) {
    hydroRun->oneStepIntegration(nStep,t,dt);
    emit dataChanged();
  } else {
    // only compute a finite number of steps
    if (nbStepToDo) {
      hydroRun->oneStepIntegration(nStep,t,dt);
      nbStepToDo--;
      emit dataChanged();
    }
  }
} // HydroWidget::updateData

// =======================================================
// =======================================================
void HydroWidget::timerEvent(QTimerEvent *)
{
  
  updateData();
  updateGL();

} // HydroWidget::timerEvent

// =======================================================
// =======================================================
void HydroWidget::createPBO()
{
  // Create pixel buffer object and bind to gl_PBO. We store the data we want to
  // plot in memory on the graphics card - in a "pixel buffer". We can then 
  // copy this to the texture defined above and send it to the screen
  
  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffersARB(1, &gl_PBO);
  
  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);

  printf("PBO created.\n");

} // HydroWidget::createPBO

// =======================================================
// =======================================================
void HydroWidget::deletePBO()
{
  if(gl_PBO) {
    
    glDeleteBuffersARB( 1, &gl_PBO );
    gl_PBO=NULL;
        
  }

  printf("PBO deleted...\n");

} // HydroWidget::deletePBO

// =======================================================
// =======================================================
void HydroWidget::createTexture()
{
  /*
   * Create texture which we use to display the result and bind to
   * gl_Tex
   */

  if(gl_Tex) deleteTexture();

  // Enable Texturing
  glEnable(GL_TEXTURE_2D);
  
  // Generate a texture identifier
  glGenTextures(1, &gl_Tex);

  // Make this the current texture (remember that GL is state-based)
  glBindTexture(GL_TEXTURE_2D, gl_Tex);

  // Allocate the texture memory. 
  // The last parameter is NULL since we only want to allocate memory,
  // not initialize it
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx+4, ny+4, 0, 
	       GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  // texture properties:
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
  // GL_TEXTURE_2D for improved performance if linear interpolation is
  // not desired. Replace GL_LINEAR with GL_NEAREST in the
  // glTexParameteri() call
  
  printf("Texture created...\n");

} // HydroWidget::createTexture

// =======================================================
// =======================================================
void HydroWidget::deleteTexture()
{  
  
  if (gl_Tex) {
    glDeleteTextures(1, &gl_Tex);
    gl_Tex = NULL;
  }

  printf("Texture deleted...\n");

} // HydroWidget::deleteTexture

// =======================================================
// =======================================================
void HydroWidget::render()
{
  // convert the plotvar array into an array of colors to plot
  
  /*
   * compute plot_rgba array and then copy that into the PBO.
   */
  // compute min / max values
  if (!manualContrast)
    computeMinMax(hydroRun->getDataHost().data(),
		  hydroRun->getDataHost().section(), 
		  displayVar);
  
  //int ncol = NBCOLORS;
  int i0;
  
  if (displayVar == 0 and minvar < 1e-2)
    minvar = 1e-2;
  
  if (minvar == maxvar)
    minvar = 0.99*maxvar;

  for (int j=0;j<ny+4;j++){
    for (int i=0;i<nx+4;i++){
      // plotvar is either density, velocity or energy
      real_t plotvar = hydroRun->getData()(i,j,displayVar);
      //real_t frac=(plotvar-minvar)/(maxvar-minvar);
      real_t frac;
      if (displayVar == 0) { 
	// use a logarithmic scale for density ??
	// does'nt seem to give so good displays
	//frac=log(maxvar/plotvar)/log(maxvar/minvar);
	frac=(maxvar-plotvar)/(maxvar-minvar);
      } else {
	frac=(maxvar-plotvar)/(maxvar-minvar);
      }
      if (frac<0.0)
	frac=0.0;
      if (frac>1.0)
	frac=1.0;
      //int icol=frac*ncol;
      i0 = I2D(nx+4,i,j);
      if (useColor) {
	//plot_rgba[i0] = cmap_rgba[icol];
	uint r,g,b;
	r = FMIN( FMAX( 4*(frac-0.25), 0.), 1.);
	g = FMIN( FMAX( 4*FABS(frac-0.5)-1., 0.), 1.);
	b = FMIN( FMAX( 4*(0.75-frac), 0.), 1.);
	
	plot_rgba[i0] = ((int)(r*255.0f) << 24) | // convert colormap to int
	  ((int)(g * 255.0f) << 16) |
	  ((int)(b * 255.0f) <<  8) |
	  ((int)(frac*255.0f) <<  0);
      } else {
	plot_rgba[i0] = ((int)(255.0f) << 24) | // convert colormap to int
	  ((int)(frac * 255.0f) << 16) |
	  ((int)(frac * 255.0f) <<  8) |
	  ((int)(frac * 255.0f) <<  0);
      }
    }
  }
  
  // Fill the pixel buffer with the plot_rgba array
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,(nx+4)*(ny+4)*sizeof(unsigned int),
		  (void **)plot_rgba,GL_STREAM_COPY);
  
} // HydroWidget::render

// =======================================================
// =======================================================
void HydroWidget::initializeGL()
{

  // Just check that we have a valid GL-context
  GL_SAFE_CALL( ; );
  
  // Check for OpenGL extension support 
  printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
  if(!glewIsSupported(
   		      "GL_VERSION_2_0 " 
   		      "GL_ARB_pixel_buffer_object "
   		      "GL_EXT_framebuffer_object "
   		      )){
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    //return;
  }

  glViewport(0, 0, nx+4, ny+4);
  
  // Set up view
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,nx+4,0.,ny+4, -200.0, 200.0);
  
  // create Texture object
  createTexture();
  
  // create Pixel Buffer object
  createPBO();

  // OpenGL info
  printf( "GL_VENDOR                   : '%s'\n", glGetString( GL_VENDOR   ) ) ;
  printf( "GL_RENDERER                 : '%s'\n", glGetString( GL_RENDERER ) ) ;
  printf( "GL_VERSION                  : '%s'\n", glGetString( GL_VERSION  ) ) ;
  printf( "GL_SHADING_LANGUAGE_VERSION : '%s'\n", 
	  glGetString ( GL_SHADING_LANGUAGE_VERSION ) );

  //glShadeModel(GL_FLAT);
  //glEnable(GL_DEPTH_TEST);
  //glEnable(GL_CULL_FACE);
}

// =======================================================
// =======================================================
void HydroWidget::paintGL()
{
  makeCurrent();

  // make the convertion and fill PBO
  render();
  
  // Copy the pixel buffer to the texture, ready to display
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,nx+4,ny+4,GL_RGBA,GL_UNSIGNED_BYTE,0);

  // Render one quad to the screen and colour it using our texture
  // i.e. plot our plotvar data to the screen
  glClear(GL_COLOR_BUFFER_BIT);
  glBegin(GL_QUADS);
  glTexCoord2f (0.0, 0.0); glVertex3f (0.0 , 0.0 , 0.0);
  glTexCoord2f (1.0, 0.0); glVertex3f (nx+4, 0.0 , 0.0);
  glTexCoord2f (1.0, 1.0); glVertex3f (nx+4, ny+4, 0.0);
  glTexCoord2f (0.0, 1.0); glVertex3f (0.0 , ny+4, 0.0);
  glEnd();


  /*
   * print time
   */
//   qglColor(Qt::blue);
//   QString timeStr;
//   timeStr.setNum(t, 'f');
//   QFont serifFont("Helvetica", 20, QFont::Bold);
//   renderText(-0.35, 0.4, 0.0, "time : "+timeStr,serifFont);
//   qglColor(Qt::white);
  
  //swapBuffers(); 

}

// =======================================================
// =======================================================
void HydroWidget::resizeGL(int width, int height)
{

  glViewport (0, 0, width, height); 
  glMatrixMode (GL_PROJECTION); 
  glLoadIdentity (); 
  glOrtho (0., nx+4, 0., ny+4, -200. ,200.); 
  glMatrixMode (GL_MODELVIEW); 
  glLoadIdentity ();

}

// =======================================================
// =======================================================
void HydroWidget::mousePressEvent(QMouseEvent *event)
{
  lastPos = event->pos();
}

// =======================================================
// =======================================================
void HydroWidget::mouseMoveEvent(QMouseEvent *event)
{
  //int dx = event->x() - lastPos.x();
  //int dy = event->y() - lastPos.y();

  if (event->buttons() & Qt::LeftButton) {
  } else if (event->buttons() & Qt::RightButton) {
  }
  lastPos = event->pos();
}

// =======================================================
// =======================================================
void HydroWidget::saveGLState()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
}

// =======================================================
// =======================================================
void HydroWidget::restoreGLState()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
}


// =======================================================
// =======================================================
void HydroWidget::computeMinMax(real_t *U, int size, int iVar)
{
  
  minvar = U[0+iVar*size];
  maxvar = U[0+iVar*size];
  for(int i=1+iVar*size; i<size+iVar*size; i++) {
    minvar = U[i]<minvar ? U[i] : minvar;
    maxvar = U[i]>maxvar ? U[i] : maxvar;
  }
  
} //HydroWidget::computeMinMax

// =======================================================
// =======================================================
/** 
 * fill host array cmap_rgba.
 * The GPU version override this routine. 
 */
void HydroWidget::initColormap() {

  float cmap[NBCOLORS][3]={
    {0.0000,    0.0000,    1.0000},
    {0.0000,    0.0157,    1.0000},
    {0.0000,    0.0314,    1.0000},
    {0.0000,    0.0471,    1.0000},
    {0.0000,    0.0627,    1.0000},
    {0.0000,    0.0784,    1.0000},
    {0.0000,    0.0941,    1.0000},
    {0.0000,    0.1098,    1.0000},
    {0.0000,    0.1255,    1.0000},
    {0.0000,    0.1412,    1.0000},
    {0.0000,    0.1569,    1.0000},
    {0.0000,    0.1725,    1.0000},
    {0.0000,    0.1882,    1.0000},
    {0.0000,    0.2039,    1.0000},
    {0.0000,    0.2196,    1.0000},
    {0.0000,    0.2353,    1.0000},
    {0.0000,    0.2510,    1.0000},
    {0.0000,    0.2667,    1.0000},
    {0.0000,    0.2824,    1.0000},
    {0.0000,    0.2980,    1.0000},
    {0.0000,    0.3137,    1.0000},
    {0.0000,    0.3294,    1.0000},
    {0.0000,    0.3451,    1.0000},
    {0.0000,    0.3608,    1.0000},
    {0.0000,    0.3765,    1.0000},
    {0.0000,    0.3922,    1.0000},
    {0.0000,    0.4078,    1.0000},
    {0.0000,    0.4235,    1.0000},
    {0.0000,    0.4392,    1.0000},
    {0.0000,    0.4549,    1.0000},
    {0.0000,    0.4706,    1.0000},
    {0.0000,    0.4863,    1.0000},
    {0.0000,    0.5020,    1.0000},
    {0.0000,    0.5176,    1.0000},
    {0.0000,    0.5333,    1.0000},
    {0.0000,    0.5490,    1.0000},
    {0.0000,    0.5647,    1.0000},
    {0.0000,    0.5804,    1.0000},
    {0.0000,    0.5961,    1.0000},
    {0.0000,    0.6118,    1.0000},
    {0.0000,    0.6275,    1.0000},
    {0.0000,    0.6431,    1.0000},
    {0.0000,    0.6588,    1.0000},
    {0.0000,    0.6745,    1.0000},
    {0.0000,    0.6902,    1.0000},
    {0.0000,    0.7059,    1.0000},
    {0.0000,    0.7216,    1.0000},
    {0.0000,    0.7373,    1.0000},
    {0.0000,    0.7529,    1.0000},
    {0.0000,    0.7686,    1.0000},
    {0.0000,    0.7843,    1.0000},
    {0.0000,    0.8000,    1.0000},
    {0.0000,    0.8157,    1.0000},
    {0.0000,    0.8314,    1.0000},
    {0.0000,    0.8471,    1.0000},
    {0.0000,    0.8627,    1.0000},
    {0.0000,    0.8784,    1.0000},
    {0.0000,    0.8941,    1.0000},
    {0.0000,    0.9098,    1.0000},
    {0.0000,    0.9255,    1.0000},
    {0.0000,    0.9412,    1.0000},
    {0.0000,    0.9569,    1.0000},
    {0.0000,    0.9725,    1.0000},
    {0.0000,    0.9882,    1.0000},
    {0.0000,    1.0000,    0.9961},
    {0.0000,    1.0000,    0.9804},
    {0.0000,    1.0000,    0.9647},
    {0.0000,    1.0000,    0.9490},
    {0.0000,    1.0000,    0.9333},
    {0.0000,    1.0000,    0.9176},
    {0.0000,    1.0000,    0.9020},
    {0.0000,    1.0000,    0.8863},
    {0.0000,    1.0000,    0.8706},
    {0.0000,    1.0000,    0.8549},
    {0.0000,    1.0000,    0.8392},
    {0.0000,    1.0000,    0.8235},
    {0.0000,    1.0000,    0.8078},
    {0.0000,    1.0000,    0.7922},
    {0.0000,    1.0000,    0.7765},
    {0.0000,    1.0000,    0.7608},
    {0.0000,    1.0000,    0.7451},
    {0.0000,    1.0000,    0.7294},
    {0.0000,    1.0000,    0.7137},
    {0.0000,    1.0000,    0.6980},
    {0.0000,    1.0000,    0.6824},
    {0.0000,    1.0000,    0.6667},
    {0.0000,    1.0000,    0.6510},
    {0.0000,    1.0000,    0.6353},
    {0.0000,    1.0000,    0.6196},
    {0.0000,    1.0000,    0.6039},
    {0.0000,    1.0000,    0.5882},
    {0.0000,    1.0000,    0.5725},
    {0.0000,    1.0000,    0.5569},
    {0.0000,    1.0000,    0.5412},
    {0.0000,    1.0000,    0.5255},
    {0.0000,    1.0000,    0.5098},
    {0.0000,    1.0000,    0.4941},
    {0.0000,    1.0000,    0.4784},
    {0.0000,    1.0000,    0.4627},
    {0.0000,    1.0000,    0.4471},
    {0.0000,    1.0000,    0.4314},
    {0.0000,    1.0000,    0.4157},
    {0.0000,    1.0000,    0.4000},
    {0.0000,    1.0000,    0.3843},
    {0.0000,    1.0000,    0.3686},
    {0.0000,    1.0000,    0.3529},
    {0.0000,    1.0000,    0.3373},
    {0.0000,    1.0000,    0.3216},
    {0.0000,    1.0000,    0.3059},
    {0.0000,    1.0000,    0.2745},
    {0.0000,    1.0000,    0.2431},
    {0.0000,    1.0000,    0.2118},
    {0.0000,    1.0000,    0.1804},
    {0.0000,    1.0000,    0.1490},
    {0.0000,    1.0000,    0.1176},
    {0.0000,    1.0000,    0.0863},
    {0.0000,    1.0000,    0.0549},
    {0.0000,    1.0000,    0.0235},
    {0.0078,    1.0000,    0.0000},
    {0.0392,    1.0000,    0.0000},
    {0.0706,    1.0000,    0.0000},
    {0.1020,    1.0000,    0.0000},
    {0.1333,    1.0000,    0.0000},
    {0.1647,    1.0000,    0.0000},
    {0.1961,    1.0000,    0.0000},
    {0.2275,    1.0000,    0.0000},
    {0.2588,    1.0000,    0.0000},
    {0.2902,    1.0000,    0.0000},
    {0.3216,    1.0000,    0.0000},
    {0.3373,    1.0000,    0.0000},
    {0.3529,    1.0000,    0.0000},
    {0.3686,    1.0000,    0.0000},
    {0.3843,    1.0000,    0.0000},
    {0.4000,    1.0000,    0.0000},
    {0.4157,    1.0000,    0.0000},
    {0.4314,    1.0000,    0.0000},
    {0.4471,    1.0000,    0.0000},
    {0.4627,    1.0000,    0.0000},
    {0.4784,    1.0000,    0.0000},
    {0.4941,    1.0000,    0.0000},
    {0.5098,    1.0000,    0.0000},
    {0.5255,    1.0000,    0.0000},
    {0.5412,    1.0000,    0.0000},
    {0.5569,    1.0000,    0.0000},
    {0.5725,    1.0000,    0.0000},
    {0.5882,    1.0000,    0.0000},
    {0.6039,    1.0000,    0.0000},
    {0.6196,    1.0000,    0.0000},
    {0.6353,    1.0000,    0.0000},
    {0.6510,    1.0000,    0.0000},
    {0.6667,    1.0000,    0.0000},
    {0.6824,    1.0000,    0.0000},
    {0.6980,    1.0000,    0.0000},
    {0.7137,    1.0000,    0.0000},
    {0.7294,    1.0000,    0.0000},
    {0.7451,    1.0000,    0.0000},
    {0.7608,    1.0000,    0.0000},
    {0.7765,    1.0000,    0.0000},
    {0.7922,    1.0000,    0.0000},
    {0.8078,    1.0000,    0.0000},
    {0.8235,    1.0000,    0.0000},
    {0.8392,    1.0000,    0.0000},
    {0.8549,    1.0000,    0.0000},
    {0.8706,    1.0000,    0.0000},
    {0.8863,    1.0000,    0.0000},
    {0.9020,    1.0000,    0.0000},
    {0.9176,    1.0000,    0.0000},
    {0.9333,    1.0000,    0.0000},
    {0.9490,    1.0000,    0.0000},
    {0.9647,    1.0000,    0.0000},
    {0.9804,    1.0000,    0.0000},
    {0.9961,    1.0000,    0.0000},
    {1.0000,    0.9882,    0.0000},
    {1.0000,    0.9725,    0.0000},
    {1.0000,    0.9569,    0.0000},
    {1.0000,    0.9412,    0.0000},
    {1.0000,    0.9255,    0.0000},
    {1.0000,    0.9098,    0.0000},
    {1.0000,    0.8941,    0.0000},
    {1.0000,    0.8784,    0.0000},
    {1.0000,    0.8627,    0.0000},
    {1.0000,    0.8471,    0.0000},
    {1.0000,    0.8314,    0.0000},
    {1.0000,    0.8157,    0.0000},
    {1.0000,    0.8000,    0.0000},
    {1.0000,    0.7843,    0.0000},
    {1.0000,    0.7686,    0.0000},
    {1.0000,    0.7529,    0.0000},
    {1.0000,    0.7373,    0.0000},
    {1.0000,    0.7216,    0.0000},
    {1.0000,    0.7059,    0.0000},
    {1.0000,    0.6902,    0.0000},
    {1.0000,    0.6745,    0.0000},
    {1.0000,    0.6588,    0.0000},
    {1.0000,    0.6431,    0.0000},
    {1.0000,    0.6275,    0.0000},
    {1.0000,    0.6118,    0.0000},
    {1.0000,    0.5961,    0.0000},
    {1.0000,    0.5804,    0.0000},
    {1.0000,    0.5647,    0.0000},
    {1.0000,    0.5490,    0.0000},
    {1.0000,    0.5333,    0.0000},
    {1.0000,    0.5176,    0.0000},
    {1.0000,    0.5020,    0.0000},
    {1.0000,    0.4863,    0.0000},
    {1.0000,    0.4706,    0.0000},
    {1.0000,    0.4549,    0.0000},
    {1.0000,    0.4392,    0.0000},
    {1.0000,    0.4235,    0.0000},
    {1.0000,    0.4078,    0.0000},
    {1.0000,    0.3922,    0.0000},
    {1.0000,    0.3765,    0.0000},
    {1.0000,    0.3608,    0.0000},
    {1.0000,    0.3451,    0.0000},
    {1.0000,    0.3294,    0.0000},
    {1.0000,    0.3137,    0.0000},
    {1.0000,    0.2980,    0.0000},
    {1.0000,    0.2824,    0.0000},
    {1.0000,    0.2667,    0.0000},
    {1.0000,    0.2510,    0.0000},
    {1.0000,    0.2353,    0.0000},
    {1.0000,    0.2196,    0.0000},
    {1.0000,    0.2039,    0.0000},
    {1.0000,    0.1882,    0.0000},
    {1.0000,    0.1725,    0.0000},
    {1.0000,    0.1569,    0.0000},
    {1.0000,    0.1412,    0.0000},
    {1.0000,    0.1255,    0.0000},
    {1.0000,    0.1098,    0.0000},
    {1.0000,    0.0941,    0.0000},
    {1.0000,    0.0784,    0.0000},
    {1.0000,    0.0627,    0.0000},
    {1.0000,    0.0471,    0.0000},
    {1.0000,    0.0314,    0.0000},
    {1.0000,    0.0157,    0.0000},
    {1.0000,    0.0000,    0.0000}};

  QGLColormap colormap;

  for (int i=0;i<NBCOLORS;i++)
    colormap.setEntry(i, qRgb(cmap[i][0], cmap[i][1], cmap[i][2]));
  
  this->setColormap(colormap);
  
} // HydroWidget::initColormap
