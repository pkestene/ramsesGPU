/**
 * \file MainWindow.h
 * \brief Defines the top level QT windows.
 *
 * \date 11-03-2010
 * \author Pierre Kestener
 */
#ifndef MAIN_WINDOW_H_
#define MAIN_WINDOW_H_

#include <QWidget>

#include <ConfigMap.h>
#include "HydroWidget.h"
#include <QGroupBox>
#include <QLabel>

/**
 * \class MainWindow MainWindow.h
 * \brief Main window gathers all composite widgets.
 */
class MainWindow : public QWidget
{
  Q_OBJECT
  
  public:
  MainWindow(ConfigMap& _param, NumScheme numScheme, HydroWidget *hydroWidget_, QWidget *parent);
  ~MainWindow();
  ConfigMap      param;
  HydroWidget   *hydroWidget;
  QGroupBox     *groupBox;
  QLabel        *infoLabel;
  
public slots:
  void updateInfoLabel();

protected:
  void keyPressEvent(QKeyEvent *event);
  
};

#endif // MAIN_WINDOW_H_
