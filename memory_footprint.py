#
# Some helper function to compute required RAM memory
# per GPU (1 GPU per MPI process)
#

# required memory space in MBytes for MHD run
# def mem_in_MB(N):
#     return (180.0*8*(N+6)*(N+6)*(N+6)+710.0*8*(N+6)*(N+6))/1000000

def mem_in_MB(Nx,Ny,Nz):
    return (180.0*8*(Nx+6)*(Ny+6)*(Nz+6)+710.0*8*(Nx+6)*(Ny+6))/1000000


# N=150 ==> 5.6 GBytes

# required memory space in MBytes for MHD run when using the zSlab method
# def mem2_in_MB(N,nPiece):
#     return (164.0*8*(N+6)*(N+6)*(N+6)/nPiece+16*8*(N+6)*(N+6)*(N+6)+710.0*8*(N+6)*(N+6))/1000000

def mem2_in_MB(Nx,Ny,Nz,nPiece):
    return (164.0*8*(Nx+6)*(Ny+6)*(Nz+6)/nPiece+16*8*(Nx+6)*(Ny+6)*(Nz+6)+710.0*8*(Nx+6)*(Ny+6))/1000000
