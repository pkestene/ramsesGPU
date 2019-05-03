# Populate a CMake variable with the sources
add_executable(qtHelloWorld "")

target_sources(qtHelloWorld
  PUBLIC
  glwidget.cpp
  window.cpp
  main.cpp
)

# Use the Widgets module from Qt 5
target_link_libraries(qtHelloWorld Qt5::Widgets Qt5::OpenGL OpenGL)
