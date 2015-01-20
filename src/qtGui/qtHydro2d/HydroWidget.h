/**
 * \file HydroWidget.h
 * \brief
 * OpenGL Widget to display HydroRun simulation (QGLWidget specialization).
 *
 * \author Pierre Kestener
 * \date 14-03-2010
 */
#ifndef HYDRO_WIDGET_H_
#define HYDRO_WIDGET_H_

// Qt headers
#include <QGLWidget>

// application header
#include <ConfigMap.h>
#include <HydroRunBase.h>

#define I2D(nx,i,j) (((nx)*(j)) + i)
#define NBCOLORS (236)

using hydroSimu::HydroRunBase;
using hydroSimu::HostArray;


/**
 * \class HydroWidget HydroWidget.h
 * \brief HydroWidget derives from QGLWidget to handle simulation
 * results display.
 */
class HydroWidget : public QGLWidget
{
  Q_OBJECT

  public:
  HydroWidget(ConfigMap& _param, HydroRunBase* _hydroRun, QWidget* parent = 0);
  virtual ~HydroWidget();

  QSize minimumSizeHint() const;
  QSize sizeHint() const;

  void startSimulation();
  void updateData();

signals:
  void plotXCut(int x, int y);
  void dataChanged();

protected:
  virtual void initializeGL();
  void paintGL();
  void resizeGL(int width, int height);
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void saveGLState();
  void restoreGLState();
  void timerEvent(QTimerEvent *);

  virtual void createPBO();
  virtual void deletePBO();
  virtual void createTexture();
  virtual void deleteTexture();
  virtual void render();

  int          nx,ny;
  // OpenGL pixel buffer object and texture identifiers
  GLuint gl_PBO, gl_Tex;

  // simulation data
  ConfigMap param;

 public:
  HydroRunBase * hydroRun;
  real_t t;
  real_t dt;
  int nStep; 
  int nbStepToDo;
  bool animate;

  int useColor;
  int manualContrast;
  int currentRiemannConfigNb;

  // colormap
  unsigned int cmap_rgba[236];
  virtual void initColormap();
  unsigned int *plot_rgba;
  real_t minvar,maxvar;
  int displayVar;
  
protected:
  QPoint lastPos;  
  virtual void computeMinMax(real_t *U, int size, int iVar);
};

#endif // GODUNOV_WIDGET_H_
