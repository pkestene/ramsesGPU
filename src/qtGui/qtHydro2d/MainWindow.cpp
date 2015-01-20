/**
 * \file MainWindow.cpp
 * \brief Implements the top level QT windows.
 *
 * \date 11-03-2010
 * \author Pierre Kestener
 */

#include <QtWidgets>

#include "MainWindow.h"

// =======================================================
// =======================================================
MainWindow::MainWindow(ConfigMap& _param, NumScheme numScheme, HydroWidget *hydroWidget_, QWidget *parent) :
  param(_param)
{

  // simulation widget 
  hydroWidget = hydroWidget_;

  // information text  label
  infoLabel     = new QLabel(this);
  infoLabel->setFrameStyle(QFrame::Panel | QFrame::Sunken);
  infoLabel->setText("first line\nsecond line");
  infoLabel->setAlignment(Qt::AlignBottom | Qt::AlignLeft);
  
  // layout 
  groupBox                 = new QGroupBox(tr("Density field"));
  QVBoxLayout *mainLayout  = new QVBoxLayout;
  QVBoxLayout *grBoxLayout = new QVBoxLayout;
  
  grBoxLayout->addWidget(hydroWidget);
  grBoxLayout->addWidget(infoLabel);
  groupBox->setLayout(grBoxLayout);
  
  mainLayout->addWidget(groupBox);
  setLayout(mainLayout);
  
  if (numScheme == GODUNOV)
    setWindowTitle(tr("Hydro 2d : Godunov scheme"));
  else if (numScheme == KURGANOV)
    setWindowTitle(tr("Hydro 2d : Kurganov-Tadmor scheme"));
  else
    setWindowTitle(tr("Hydro 2d : Relaxing TVD scheme"));

  // connect signals
  connect(hydroWidget, SIGNAL(dataChanged()), this, SLOT(updateInfoLabel()));

}

// =======================================================
// =======================================================
MainWindow::~MainWindow()
{

  delete infoLabel;

  delete groupBox;
  
}

// =======================================================
// =======================================================
void MainWindow::updateInfoLabel()
{
  QString timeStr;
  timeStr.setNum(hydroWidget->t, 'f');
  QString nStepStr;
  nStepStr.setNum(hydroWidget->nStep);
  infoLabel->setText("time : "+timeStr+" (nstep = "+nStepStr+")");
} // MainWindow::updateInfoLabel

// =======================================================
// =======================================================
void MainWindow::keyPressEvent(QKeyEvent *e)
{
  switch (e->key()) {
  case Qt::Key_R : /* reset simulation */
    {
      hydroWidget->t=0.0f;
      hydroWidget->nStep=0;
      std::string problem = param.getString("hydro", "problem", "unknown");
      if (!problem.compare("jet")) {
	hydroWidget->hydroRun->init_hydro_jet();
      } else if (!problem.compare("implode")) {
	hydroWidget->hydroRun->init_hydro_implode();
      } else if (!problem.compare("blast")) {
	hydroWidget->hydroRun->init_hydro_blast();
      } else if (!problem.compare("Kelvin-Helmholtz")) {
	hydroWidget->hydroRun->init_hydro_Kelvin_Helmholtz();
      } else if (!problem.compare("Rayleigh-Taylor")) {
	hydroWidget->hydroRun->init_hydro_Rayleigh_Taylor();
      } else if (!problem.compare("riemann2d")) {
	if (e->modifiers() & Qt::ShiftModifier) {
	  /* change initial condition in case of the Riemann 2D problem */
	  /* just increment Riemann config number */
	  if (hydroWidget->currentRiemannConfigNb == 19)
	    hydroWidget->currentRiemannConfigNb = 0;
	  else
	    hydroWidget->currentRiemannConfigNb++;
	  printf("current Riemann config number : %d\n",hydroWidget->currentRiemannConfigNb);
	}
	hydroWidget->hydroRun->setRiemannConfId(hydroWidget->currentRiemannConfigNb);
	hydroWidget->hydroRun->init_hydro_Riemann();
      }
    }
    break;

  case Qt::Key_Escape : case Qt::Key_Q :
    close();
    printf("Bye !!!\n");
    break;

  case Qt::Key_C :
    hydroWidget->useColor=1-hydroWidget->useColor;
    break;
    
  case Qt::Key_D : /* dump simulation results in HDF5/XDMF
		      format */
    {
      if (!hydroWidget->animate) // dump is only activated when simulations
	// is stopped
	{
	  // write heavy data in HDF5
	  hydroWidget->hydroRun->outputHdf5(hydroWidget->hydroRun->getDataHost(),hydroWidget->nStep);
	  // write light data in XDMF for a single step (usefull for
	  // paraview loading)
	  hydroWidget->hydroRun->writeXdmfForHdf5Wrapper(hydroWidget->nStep,true);
	  std::cout << "dump results into HDF5/XDMF file for step : " << hydroWidget->nStep << std::endl;
	}
    }
    break;
  
  case Qt::Key_Space : /* Start/Stop the animation */
    hydroWidget->animate = not hydroWidget->animate;
    if (hydroWidget->animate)
      std::cout << "restarting simulations from step : " << hydroWidget->nStep << std::endl;
    break;

  case Qt::Key_S : /* Do only a single step */
    {
      std::cout << "current nStep : " << hydroWidget->nStep 
		<< "\t time : " << hydroWidget->t << std::endl;
      hydroWidget->animate=false;
      hydroWidget->nbStepToDo=1;
    }
    break;

  case Qt::Key_A :
    {
      if (e->modifiers() & Qt::ShiftModifier) {
      	hydroWidget->maxvar -=0.1;
      } else {
	hydroWidget->maxvar +=0.1;
      }
      printf("maxvar (current value): %f\n",hydroWidget->maxvar);
    }
    break;

  case Qt::Key_B :
    {
      if (e->modifiers() & Qt::ShiftModifier) {
	hydroWidget->minvar -=0.1;
      } else {
	hydroWidget->minvar +=0.1;
      }
      printf("minvar (current value): %f\n",hydroWidget->minvar);
    }
    break;

  case Qt::Key_U :
    {
      hydroWidget->displayVar++;
      if (hydroWidget->displayVar==4)
	hydroWidget->displayVar=0;
      switch (hydroWidget->displayVar)
	{
	case 0:
	  printf("display Density\n");
	  groupBox->setTitle("Density field");
	  break;
	case 1:
	  printf("display X velocity\n");
	  groupBox->setTitle("X-velocity field");
	  break;
	case 2:
	  printf("display Y velocity\n");
	  groupBox->setTitle("Y-velocity field");
	  break;
	case 3:
	  printf("display Energy\n");
	  groupBox->setTitle("Energy field");
	  break;
	}
    }
    break;

  default:
    QWidget::keyPressEvent(e);
  }

}
