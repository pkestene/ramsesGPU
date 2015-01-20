/**
 * \file testVtkXMLPImageDataWriter.cpp
 * \brief Example of use of VTK class vtkXMLPImageDataWriter with MPI.
 *
 * In this example a 2D ImageData object is created and then cut into
 * pieces which are written by different MPI processes into multiple
 * vti files.
 * argv[1] must be a filename with extension pvti
 * argv[2] must be the total number of pieces.
 *
 * This example is adapted from the code found on the following web
 * page:
 * http://vtk.1045678.n5.nabble.com/MPI-pvti-num-pieces-num-procs-tt1228683.html#a1228683
 *
 *
 * Important note: 
 * The pvti file format expects that, even when the ghost level is zero, 
 * all adjacent pieces overlap by one cell !!! 
 * 
 * Just try to execute this example :
 *     mpirun -n 4 ./testVtkXMLPImageDataWriter test.pvti 4
 * You will notice that extents of piece test_0.vti is "0 99 0 49 0 0" and 
 * extent of piece test_1.vti is "0 99 49 99 0 0"; that means that data at 
 * y=49 will be written twice in these two files !!!
 *
 *
 *
 * \date 4 Oct 2010
 * \author P. Kestener
 *
 */

#define VTK_EXCLUDE_STRSTREAM_HEADERS

#include <vtkMPIController.h>

#include <vtkSmartPointer.h>
#include <vtkImageData.h>

#include <vtkImageWriter.h>
#include <vtkXMLPImageDataWriter.h>

#define NX 100
#define NY 200
#define NZ   1

struct args_tmp 
{ 
  int argc; 
  char** argv; 
}; 

/**
 * entry point of the code run by each MPI process.
 */
void process(vtkMultiProcessController* controller, void* arg) 
{ 
  int myId = controller->GetLocalProcessId(); 
  int numProcs = controller->GetNumberOfProcesses(); 

  args_tmp * args_ptr 
    = reinterpret_cast< args_tmp * >(arg); 

  char* out_file_name = args_ptr->argv[1]; 
  int pieces = atoi( args_ptr->argv[2] ); 

  // Create ImageData object and fill with dummy data
  vtkSmartPointer<vtkImageData> imageData = 
    vtkSmartPointer<vtkImageData>::New();
#if HAVE_VTK6
  imageData->SetExtent(0,NX-1,0,NY-1,0,NZ-1);
#else
  imageData->SetDimensions(NX, NY, NZ);
#endif 

  imageData->SetOrigin(0.0, 0.0, 0.0);
  imageData->SetSpacing(1.0,1.0,1.0);

#if HAVE_VTK6
  imageData->AllocateScalars(VTK_FLOAT, 3);
#else
  imageData->SetNumberOfScalarComponents(1);
  imageData->SetScalarTypeToFloat();
  imageData->AllocateScalars();
#endif
  for(int j= 0; j < NY; j++)
    for(int i = 0; i < NX; i++) {
      float* tmp = static_cast<float*>( imageData->GetScalarPointer(i,j,0) );
      tmp[0] = i+j;
    }

  // compute number of pieces per MPI process
  int pieces_per_node = pieces / numProcs; 
  int rem_pieces = pieces % numProcs; 
  int start_piece; 
  int end_piece; 
  start_piece = myId * pieces_per_node; 
  end_piece = (myId + 1) * pieces_per_node - 1; 
  if ( myId < rem_pieces ) 
    { 
      start_piece += myId; 
      end_piece += myId; 
    } 
  else 
    { 
      start_piece += rem_pieces; 
      end_piece += rem_pieces; 
    } 
  std::cout << "Process "; 
  std::cout << myId; 
  std::cout << " will write pieces "; 
  std::cout << start_piece << " - "; 
  std::cout << end_piece << std::endl; 

  // create the parallel writer
  vtkSmartPointer<vtkXMLPImageDataWriter> image_writer 
    = vtkSmartPointer<vtkXMLPImageDataWriter>::New();

  image_writer->SetFileName(out_file_name ); 
  image_writer->SetNumberOfPieces( pieces ); 
#if HAVE_VTK6
  image_writer->SetInputData( imageData );
#else
  image_writer->SetInput( imageData );
#endif
  image_writer->SetByteOrderToLittleEndian();
  image_writer->SetStartPiece( start_piece ); 
  image_writer->SetEndPiece( end_piece );      
  image_writer->SetGhostLevel(0);
  image_writer->SetDataModeToAscii();
  /* SetCompressorTypeToNone  is only available in vtk > 5.2 */
  //image_writer->SetCompressorTypeToNone();
  image_writer->SetCompressor(NULL);
  image_writer->Write();
} 

/*********************************
 *********************************/
int main( int argc, char* argv[] ) 
{ 
        
  vtkMPIController* controller = vtkMPIController::New(); 
  controller->Initialize(&argc, &argv); 

  if (argc<3) {
    if (controller->GetLocalProcessId() == 0) {
      std::cerr << "Usage:\n";
      std::cerr << "  mpirun -n nMpiProcs ./testVtkXMLPImageDataWriter filename.pvti nbOfPieces\n";
    }
    controller->Finalize(); 
    controller->Delete(); 
    return 1;
  }

  args_tmp args; 
  args.argc = argc; 
  args.argv = argv; 

  controller->SetSingleMethod(process, &args); 
  controller->SingleMethodExecute(); 

  controller->Finalize(); 
  controller->Delete(); 
      
  return 0; 
} 

