/**
 * \file HydroWindow.cpp
 * \brief Implementation of class HydroWindow.
 *
 * \author P. Kestener
 * \date 23/02/2010
 *
 * $Id: HydroWindow.cpp 1784 2012-02-21 10:34:58Z pkestene $
 */
#include <HydroWindow.h>

#include "palettes.h"

#include "../utils/monitoring/date.h"

// =======================================================
// =======================================================
HydroWindow::HydroWindow(HydroRunBase* _hydroRun,
			 GlutMaster * glutMaster,
			 int _initPositionX, int _initPositionY,
			 const char* title,
			 ConfigMap & _param) :
  nx(0), ny(0), ghostWidth(2),
  nxg(0), nyg(0),
  initPositionX(_initPositionX), initPositionY(_initPositionY), 
  ghostIncluded(false),
  gl_PBO(0), gl_Tex(0), 
  param(_param), hydroRun(_hydroRun), 
  t(0.0), dt(0.0), nStep(0), nbStepToDo(0), animate(true), 
  useColor(0), manualContrast(0),
  currentRiemannConfigNb(0),
  minvar(0), maxvar(0),
  minmaxBlockCount(192)
#ifdef __CUDACC__
  ,h_minmax()
  ,d_minmax()
#endif // __CUDACC__
{

#ifdef __CUDACC__
# ifdef USE_CUDA3
  cuda_PBO = NULL;
# endif // USE_CUDA3
#endif // __CUDACC__


  nx     = param.getInteger("mesh", "nx", 100);
  ny     = param.getInteger("mesh", "ny", 100);
  ghostWidth = param.getInteger("mesh", "ghostWidth", 2);
  bool mhdEnabled = param.getBool("MHD","enable", false);
  if (mhdEnabled) {
    ghostWidth = 3;
  }
  nxg    = nx+2*ghostWidth;
  nyg    = ny+2*ghostWidth;
  ghostIncluded = param.getBool("output","ghostIncluded",false);
  
  currentRiemannConfigNb = param.getInteger("hydro", "riemann_config_number", 0);

  /*
   * set initial conditions
   */
  std::cout << "Init Hydro2d run\n";
  std::string problem = param.getString("hydro", "problem", "unknown");
  hydroRun->setRiemannConfId(currentRiemannConfigNb);
  hydroRun->init_simulation(problem);

  // ensure boundary conditions are OK at time=0
  hydroRun->make_boundaries(hydroRun->getData(0),XDIR);
  hydroRun->make_boundaries(hydroRun->getData(0),YDIR);

  // if using the unsplit scheme, results hold in h_U (resp d_U) or
  // h_U2 (resp d_U2) according to nStep parity
  unsplitEnabled = param.getBool("hydro","unsplit", true);
  if (param.getBool("MHD","enable",false))
    unsplitEnabled = true;

#ifdef __CUDACC__
  minmaxBlockCount = std::min(minmaxBlockCount, blocksFor(hydroRun->getData(0).section(), MINMAX_BLOCK_SIZE * 2));
  printf("minmaxBlockCount %d\n",minmaxBlockCount);
  h_minmax.allocate(make_uint3(minmaxBlockCount, 1, 1));
  d_minmax.allocate(make_uint3(minmaxBlockCount, 1, 1));
#endif // __CUDACC__

  // init color map
  std::cout << "init colormap" << std::endl;
  cmap     = new float[3*NBCOLORS];
  cmap_der = new float[3*NBCOLORS];

  initColormap();
  plot_rgba = new unsigned int[nxg*nyg];
  minvar = param.getFloat("visu", "minvar", 0.0f);
  maxvar = param.getFloat("visu", "maxvar", 0.2f);
  manualContrast = param.getInteger("visu", "manualContrast", 0);
  displayVar = 0;
  computeMinMax(hydroRun->getDataHost(0),
		hydroRun->getDataHost(0).section(), 
		displayVar);
  //minvar=0.0;
  //maxvar=2.0;
  printf("constructor : min %f max %f\n",minvar,maxvar);

  /*
   * OpenGL glut : create window
   */
  printf("Glut : create window of size %d %d (%d %d)...\n",nx, ny, initPositionX, initPositionY);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(nxg,nyg);
  glutInitWindowPosition(initPositionX, initPositionY);
  glViewport(0, 0, nxg, nyg);

  glutMaster->CallGlutCreateWindow(title, this);
  
  /*
   * Check for OpenGL extension support 
   */
  printf("Loading GLEW extensions: %s\n", glewGetErrorString(glewInit()));
  if(!glewIsSupported(
   		      "GL_VERSION_2_0 " 
   		      "GL_ARB_pixel_buffer_object "
   		      "GL_EXT_framebuffer_object "
   		      )){
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    //return;
  }
  
  // Set up view (necessary : don't know)
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0., nxg, 0., nyg, -200.0, 200.0);
  
  glutReportErrors();

  createTexture();
  createPBO();

  // read animation param from config
  animate = param.getBool("output", "animate", true);

} // HydroWindow::HydroWindow

// =======================================================
// =======================================================
HydroWindow::~HydroWindow()
{

  delete [] plot_rgba;
  delete [] cmap;
  delete [] cmap_der;

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaFree(d_cmap)     );
  CUDA_SAFE_CALL( cudaFree(d_cmap_der) );
#endif // __CUDACC__

  deletePBO();
  deleteTexture();

  glutDestroyWindow(windowID);

} // HydroWindow::~HydroWindow

// =======================================================
// =======================================================
void HydroWindow::createPBO()
{

  // Create pixel buffer object and bind to gl_PBO. We store the data we want to
  // plot in memory on the graphics card - in a "pixel buffer". We can then 
  // copy this to the texture defined above and send it to the screen

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffersARB(1,&gl_PBO);
  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);

#ifdef __CUDACC__
  // Reserve memory space for PBO data (see plot_rgba)
  //glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, (nxg)*(nyg)*sizeof(unsigned int), NULL, GL_STREAM_COPY_ARB);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, hydroRun->getData(0).sectionBytes(), NULL, GL_STREAM_COPY_ARB);

  // Is the following line really necessary ???
  //glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

#  ifdef USE_CUDA3
  CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &cuda_PBO, gl_PBO, cudaGraphicsMapFlagsNone ) );
  cutilCheckMsg( "cudaGraphicsGLRegisterBuffer failed");
#  else
  CUDA_SAFE_CALL( cudaGLRegisterBufferObject( gl_PBO ) );
  cutilCheckMsg( "cudaGLRegisterBufferObject failed");
#  endif // USE_CUDA3
#endif // __CUDACC__  
  
  printf("PBO created...\n");
  
} // HydroWindow::createPBO

// =======================================================
// =======================================================
void HydroWindow::deletePBO()
{

  if(gl_PBO) {
    // delete the gl_PBO
#ifdef __CUDACC__
# ifdef USE_CUDA3
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( cuda_PBO ) );
    cutilCheckMsg( "cudaGraphicsUnRegisterResource failed");
# else
    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( gl_PBO ) );
    cutilCheckMsg( "cudaGLUnRegisterBufferObject failed");
# endif // USE_CUDA3
#endif // __CUDACC__
    
    glDeleteBuffersARB( 1, &gl_PBO );
    gl_PBO=NULL;
    
#ifdef __CUDACC__
# ifdef USE_CUDA3
    cuda_PBO = NULL;
# endif // USE_CUDA3
#endif // __CUDACC__
    
  }

  printf("PBO deleted...");
  
} // HydroWindow::deletePBO

// =======================================================
// =======================================================
void HydroWindow::createTexture()
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
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nxg, nyg, 0, 
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

} // HydroWindow::createTexture

// =======================================================
// =======================================================
void HydroWindow::deleteTexture()
{
  if (gl_Tex) {
    glDeleteTextures(1, &gl_Tex);
    gl_Tex = NULL;
  }

  printf("Texture deleted...");
  
} // HydroWindow::deleteTexture

// =======================================================
// =======================================================
void HydroWindow::render(int bufferChoice) {

  /*
   * convert the plotvar array into an array of colors to plot
   * The CUDA version is done by a kernel.
   */

#ifdef __CUDACC__
  CUDA_SAFE_THREAD_SYNC( );

  // For plotting, map the gl_PBO pixel buffer into CUDA context
  // space, so that CUDA can modify the device pointer plot_rgba_pbo
#ifdef USE_CUDA3
  CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &cuda_PBO, NULL));
  cutilCheckMsg( "cudaGraphicsMapResources failed");

  size_t num_bytes; 
  CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void **)&plot_rgba_pbo, &num_bytes, cuda_PBO));
  cutilCheckMsg( "cudaGraphicsResourceGetMappedPointer failed");
#else
  CUDA_SAFE_CALL( cudaGLMapBufferObject((void**)&plot_rgba_pbo, gl_PBO) );
#endif // USE_CUDA3
    
  // Fill the plot_rgba_data array (and thus the pixel buffer)
  convertDataForPlotting(useColor, bufferChoice);
    
  // unmap the PBO, so that OpenGL can safely do whatever he wants
#ifdef USE_CUDA3
  CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &cuda_PBO, NULL));
  cutilCheckMsg( "cudaGraphicsUnmapResources failed" );
#else
  CUDA_SAFE_CALL( cudaGLUnmapBufferObject(gl_PBO) );
  cutilCheckMsg( "cudaGLUnmapBufferObject failed" );
#endif // USE_CUDA3

#else // CPU version

  /*
   * compute plot_rgba array and then copy that into the PBO.
   */

  // compute min / max values
  if (!manualContrast)
    computeMinMax(hydroRun->getDataHost(bufferChoice),
		  hydroRun->getDataHost(bufferChoice).section(), 
		  displayVar);
    
  //printf("min-max: %f %f\n",minvar,maxvar);

  int i0;
  for (int j=0;j<nyg;j++){
    for (int i=0;i<nxg;i++){

      real_t plotvar = hydroRun->getData(bufferChoice)(i,j,displayVar);
      //real_t frac=(plotvar-minvar)/(maxvar-minvar);
      real_t frac=(maxvar-plotvar)/(maxvar-minvar);

      if (frac<0.0)
	frac=0.0;
      if (frac>1.0)
	frac=1.0;

      frac *= 255.0;

      i0 = I2D(nxg,i,j);

      if (useColor) {
	int r,g,b;
	uint iCol = (uint) (frac);
	r = (int) ( (cmap[3*iCol  ]  + (frac-iCol)*cmap_der[3*iCol  ] ) * 255.0 );
	g = (int) ( (cmap[3*iCol+1]  + (frac-iCol)*cmap_der[3*iCol+1] ) * 255.0 );
	b = (int) ( (cmap[3*iCol+2]  + (frac-iCol)*cmap_der[3*iCol+2] ) * 255.0 );
  
	plot_rgba[i0] = (r << 24) | // convert colormap to int
	  (g    << 16) |
	  (b    <<  8) |
	  ((int) (frac) <<  0);
      } else {
	plot_rgba[i0] =  
	  ((int)(255.0f) << 24) | // convert colormap to int
	  ((int)(frac) << 16) |
	  ((int)(frac) <<  8) |
	  ((int)(frac) <<  0);
      }
    }
  }
    
  // Fill the pixel buffer object with the plot_rgba array
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,(nxg)*(nyg)*sizeof(unsigned int),
	       (void **)plot_rgba,GL_STREAM_COPY);
#endif // __CUDACC__

} // HydroWindow::render

// =======================================================
// =======================================================
void HydroWindow::CallBackDisplayFunc(void){

  if (animate) {

    // compute and display continuously
    computeOneStepAndDisplay();

  } else {
    
    // only compute/display a finite number of steps
    if (nbStepToDo) {
      computeOneStepAndDisplay();
      nbStepToDo--;
    }
    
  }

} // HydroWindow::CallBackDisplayFunc

// =======================================================
// =======================================================
void HydroWindow::CallBackReshapeFunc(int w, int h){

  glViewport (0, 0, w, h); 
  glMatrixMode (GL_PROJECTION); 
  glLoadIdentity ();

  glOrtho (0., nxg, 0., nyg, -200. ,200.); 
  glMatrixMode (GL_MODELVIEW); 
  glLoadIdentity ();
  
  //glViewport(0, 0, width, height); 
  CallBackDisplayFunc();
  
} // HydroWindow::CallBackReshapeFunc

// =======================================================
// =======================================================
void HydroWindow::CallBackIdleFunc(void) {

  CallBackDisplayFunc();

} // HydroWindow::CallBackIdleFunc

// =======================================================
// =======================================================
void HydroWindow::CallBackKeyboardFunc(unsigned char key, int x, int y) {

  (void) x;
  (void) y;
  switch ( key )
    {
    case 'r' : /* Reset the simulation */
      {
	t=0.0f;
	nStep=0;
	std::string problem = param.getString("hydro", "problem", "unknown");
	hydroRun->init_simulation(problem);
      }
      break;
      
    case 'R' : /* Reset simulation and change initial condition in case 
		* of the Riemann 2D problem */
      {
	t=0.0f;
	nStep=0;
	std::string problem = param.getString("hydro", "problem", "unknown");

	// if using the riemann2d init condition, increment Riemann config number
	if (!problem.compare("riemann2d")) {
	  if (currentRiemannConfigNb == 19)
	    currentRiemannConfigNb = 0;
	  else
	    currentRiemannConfigNb++;
	  printf("current Riemann config number : %d\n",currentRiemannConfigNb);
	}
	hydroRun->setRiemannConfId(currentRiemannConfigNb);
	hydroRun->init_simulation(problem);

	printf("min max : %f %f\n",minvar,maxvar);
      }
      break;
      
    case 'c' : case 'C' :
      useColor=1-useColor;
      break;
      
    case 'd' : case 'D' : /* dump simulation results in HDF5/XDMF
			     format */
      {
	if (!animate) // dump is only activated when simulations
	              // is stopped
	  {
	    // make sure Device data are copied back onto Host memory
	    hydroRun->copyGpuToCpu(nStep);
	    
	    // write heavy data in HDF5
	    hydroRun->outputHdf5(hydroRun->getDataHost(),nStep, ghostIncluded);
	    // write light data in XDMF for a single step (usefull for
	    // paraview loading)
	    hydroRun->writeXdmfForHdf5Wrapper(nStep,true, ghostIncluded);
	    std::cout << "dump results into HDF5/XDMF file for step : " << nStep << std::endl;
	  }
      }
      break;
      
    case 32 : /* keyboard SPACE: Start/Stop the animation */
      animate= not animate;
      if (animate) {
	std::cout << "restarting simulations from step : " << nStep << std::endl;
      }
      break ;
      
    case 's' : case 'S' : /* Do only a single step */
      {
	std::cout << "[" << current_date() << "]"
		  << "  nStep : " << nStep 
		  << "\t time : " << t << " (dt=" << dt << ")" << std::endl;
	animate=false;
	nbStepToDo=1;
      }
      break;

    case 'a' : /* + */
      maxvar +=0.1;
      printf("maxvar (current value): %f\n",maxvar);
      break;
      
    case 'A' : /* - */
      maxvar -=0.1;
      printf("maxvar (current value): %f\n",maxvar);
      break;
      
    case 'b' : /* + */
      minvar +=0.1;
      printf("minvar (current value): %f\n",minvar);
      break;
      
    case 'B' : /* + */
      minvar -=0.1;
      printf("minvar (current value): %f\n",minvar);
      break;
      
    case 'u' :
      {
	displayVar++;
	if (displayVar==hydroRun->nbVar)
	  displayVar=0;
	switch (displayVar)
	  {
	  case ID:
	    printf("display Density\n");
	    break;
	  case IP:
	    printf("display Energy\n");
	    break;
	  case IU:
	    printf("display X velocity\n");
	    break;
	  case IV:
	    printf("display Y velocity\n");
	    break;
	    /* only for MHD : */
	  case IW:
	    printf("display Z velocity\n");
	    break;
	  case IA:
	    printf("display X magnetic field\n");
	    break;
	  case IB:
	    printf("display Y magnetic field\n");
	    break;
	  case IC:
	    printf("display Z magnetic field\n");
	    break;
	  }
      }
      break;
      
    case 'q' : case 27 : /* Escape key */
      glutLeaveMainLoop () ;
      break ;
    }
  
} // HydroWindow::CallBackKeyboardFunc

// =======================================================
// =======================================================
void HydroWindow::startSimulation(GlutMaster * glutMaster){

  glutMaster->SetIdleToCurrentWindow();
  glutMaster->EnableIdleFunction();

} // HydroWindow::startSimulation
   
// =======================================================
// =======================================================
void HydroWindow::computeOneStepAndDisplay()
{

  // do one Hydro step:
  hydroRun->oneStepIntegration(nStep, t, dt);
  
  // fill the PBO
  if (unsplitEnabled)
    render(nStep % 2);
  else
    render(0);
  
  // Copy the pixel buffer to the texture, ready to display
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,nxg,nyg,GL_RGBA,GL_UNSIGNED_BYTE,0);
  
  // Render one quad to the screen and colour it using our texture
  // i.e. plot our plotvar data to the screen
  glClear(GL_COLOR_BUFFER_BIT);
  glBegin(GL_QUADS);
  glTexCoord2f (0.0, 0.0); glVertex3f (0.0 , 0.0 , 0.0);
  glTexCoord2f (1.0, 0.0); glVertex3f (nxg , 0.0 , 0.0);
  glTexCoord2f (1.0, 1.0); glVertex3f (nxg , nyg , 0.0);
  glTexCoord2f (0.0, 1.0); glVertex3f (0.0 , nyg , 0.0);
  glEnd();
  
  glutSwapBuffers();
  glutReportErrors();
  
} // HydroWindow::computeOneStepAndDisplay


// =======================================================
// =======================================================
/** 
 * Set colormap and compute colormap derivatives.
 *
 * The GPU version also copy these arrays to device memory for use in
 * routine convertDataForPlotting
 * 
 */
void HydroWindow::initColormap() {

  std::string colormapName = param.getString("visu", "colormap", "rainbow");

  if (!colormapName.compare("rainbow")) {
    cmap = palette_rgb[RAINBOW];
  } else if (!colormapName.compare("jh_colors")) {
    cmap = palette_rgb[JH_COLORS];
  } else if (!colormapName.compare("step8")) {
    cmap = palette_rgb[STEP8];
  } else if (!colormapName.compare("step32")) {
    cmap = palette_rgb[STEP32];
  } else if (!colormapName.compare("idl1")) {
    cmap = palette_rgb[IDL1];
  } else if (!colormapName.compare("idl2")) {
    cmap = palette_rgb[IDL2];
  } else if (!colormapName.compare("heat")) {
    cmap = palette_rgb[HEAT];
  } else {
    cmap = palette_rgb[HEAT];
  }

  // compute derivatives (for color interpolation)
  for (int i=0;i<NBCOLORS-1;i++){
    cmap_der[3*i  ] = cmap[3*(i+1)  ] - cmap[3*i  ];
    cmap_der[3*i+1] = cmap[3*(i+1)+1] - cmap[3*i+1];
    cmap_der[3*i+2] = cmap[3*(i+1)+2] - cmap[3*i+2];
  }

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMalloc((void **)&d_cmap, 
			     sizeof(float)*NBCOLORS*3) );
  CUDA_SAFE_CALL( cudaMemcpy((void *)d_cmap,
			     (void *)cmap, sizeof(float)*NBCOLORS*3,
			     cudaMemcpyHostToDevice) );

  CUDA_SAFE_CALL( cudaMalloc((void **)&d_cmap_der, 
			     sizeof(float)*NBCOLORS*3) );
  CUDA_SAFE_CALL( cudaMemcpy((void *)d_cmap_der,
			     (void *)cmap_der, sizeof(float)*NBCOLORS*3,
			     cudaMemcpyHostToDevice) );
#endif // __CUDACC__

} // HydroWindow::initColormap


// =======================================================
// =======================================================
#ifdef __CUDACC__
/** 
 * this is a wrapper to call the CUDA kernel which actually convert data
 * from h_U to the pixel buffer object.
 * @param _useColor : switch between colormap or greymap 
 */
void HydroWindow::convertDataForPlotting(int _useColor, int bufferChoice) {

  // first compute Min / Max values to properly handle contrast
  if (!manualContrast)
    computeMinMax(hydroRun->getData(bufferChoice),
		  hydroRun->getData(bufferChoice).section(),
		  displayVar);
  
  //printf("min-max: %f %f\n",minvar,maxvar);

  dim3 grid = dim3(blocksFor(nxg,PBO_BLOCK_DIMX), blocksFor(nyg,PBO_BLOCK_DIMY));
  dim3 block = dim3(PBO_BLOCK_DIMX, PBO_BLOCK_DIMY);
  
  if (_useColor) {
    conversion_rgba_kernel<1><<<grid, block>>>(hydroRun->getData(bufferChoice).data(), 
  					       plot_rgba_pbo,
  					       d_cmap,
  					       d_cmap_der,
  					       NBCOLORS, 
  					       hydroRun->getData().pitch(),
  					       hydroRun->getData().dimx(), 
  					       hydroRun->getData().dimy(), 
  					       minvar, maxvar,
  					       displayVar);  
  } else {
    conversion_rgba_kernel<0><<<grid, block>>>(hydroRun->getData(bufferChoice).data(), 
    					       plot_rgba_pbo,
    					       d_cmap,
					       d_cmap_der,
  					       NBCOLORS, 
  					       hydroRun->getData().pitch(),
  					       hydroRun->getData().dimx(),
  					       hydroRun->getData().dimy(),
  					       minvar, maxvar,
  					       displayVar);    
  }
  CUT_CHECK_ERROR("kernel conversion_rgba_kernel failed.");

} //HydroWindow::convertDataForPlotting 
#endif // __CUDACC__

// =======================================================
// =======================================================
void HydroWindow::computeMinMax(HostArray<real_t> &U, int size, int iVar)
{
  
  real_t *data = U.data();

  minvar = data[0+iVar*size];
  maxvar = data[0+iVar*size];
  for(int i=1+iVar*size; i<size+iVar*size; i++) {
    minvar = data[i]<minvar ? data[i] : minvar;
    maxvar = data[i]>maxvar ? data[i] : maxvar;
  }
 
} // HydroWindow::computeMinMax (CPU version)


#ifdef __CUDACC__
// =======================================================
// =======================================================
void HydroWindow::computeMinMax(DeviceArray<real_t> &U, int size, int iVar)
{
  minmax_kernel<MINMAX_BLOCK_SIZE><<<
    minmaxBlockCount, 
    MINMAX_BLOCK_SIZE, 
    MINMAX_BLOCK_SIZE*sizeof(real2_t)>>>(U.data(),
					 d_minmax.data(), 
					 hydroRun->getData().section(),
					 hydroRun->getData().pitch(),
					 hydroRun->getData().dimx(),
					 iVar);
  d_minmax.copyToHost(h_minmax);
  real2_t* minmax = h_minmax.data();
  
  maxvar = -3.40282347e+38f;
  minvar =  3.40282347e+38f;

  for(uint i = 0; i < minmaxBlockCount; ++i)
    {
      minvar = FMIN(minvar, minmax[i].x);
      maxvar = FMAX(maxvar, minmax[i].y);
      //if (i%100==0) printf("minmax[%d].x=%f minmax[%d].y=%f\n",i,minmax[i].x,i,minmax[i].y);
    }

} // HydroWindow::computeMinMax (GPU version)
#endif // __CUDACC__
