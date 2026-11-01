#include "real_type.h"
#include "constants.h"

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#  define UNSPLIT_BLOCK_DIMX_2D_V4 16
#  define UNSPLIT_BLOCK_INNER_DIMX_2D_V4 (UNSPLIT_BLOCK_DIMX_2D_V4 - 4)
#  define UNSPLIT_BLOCK_DIMY_2D_V4 12
#  define UNSPLIT_BLOCK_INNER_DIMY_2D_V4 (UNSPLIT_BLOCK_DIMY_2D_V4 - 4)
#else // simple precision
#  define UNSPLIT_BLOCK_DIMX_2D_V4 16
#  define UNSPLIT_BLOCK_INNER_DIMX_2D_V4 (UNSPLIT_BLOCK_DIMX_2D_V4 - 4)
#  define UNSPLIT_BLOCK_DIMY_2D_V4 24
#  define UNSPLIT_BLOCK_INNER_DIMY_2D_V4 (UNSPLIT_BLOCK_DIMY_2D_V4 - 4)
#endif // USE_DOUBLE

/**
 * Primitive variables array (for shared memory storage).
 */
class QShmem
{
private:
  real_t data[UNSPLIT_BLOCK_DIMX_2D_V4 * UNSPLIT_BLOCK_DIMY_2D_V4 * NVAR_2D];

  static const int DIM_XY = UNSPLIT_BLOCK_DIMX_2D_V4 * UNSPLIT_BLOCK_DIMY_2D_V4;

public:
  __device__ void
  set(int const tx, int const ty, int const i_var, real_t const value)
  {
    // index = tx + UNSPLIT_BLOCK_DIMX_2D_V4 * ty +
    data[i_var + UNSPLIT_BLOCK_DIMY_2D_V4 * ty + DIM_XY * i_var] = value;
  }

  __device__ real_t
  get(int const tx, int const ty, int const i_var) const
  {
    return data[i_var + UNSPLIT_BLOCK_DIMY_2D_V4 * ty + DIM_XY * i_var];
  }
};

/**
 * Reconstructed states array (for shared memory storage).
 */
class QReconsShmem
{
private:
  static const int DIM_X1 = UNSPLIT_BLOCK_INNER_DIMX_2D_V4 + 2;
  static const int DIM_Y1 = UNSPLIT_BLOCK_INNER_DIMY_2D_V4 + 0;
  static const int DIM_XY1 = DIM_X1 * DIM_Y1;

  static const int DIM_X2 = UNSPLIT_BLOCK_INNER_DIMX_2D_V4 + 0;
  static const int DIM_Y2 = UNSPLIT_BLOCK_INNER_DIMY_2D_V4 + 2;
  static const int DIM_XY2 = DIM_X2 * DIM_Y2;

  real_t data[(UNSPLIT_BLOCK_INNER_DIMX_2D_V4 + 2) * UNSPLIT_BLOCK_DIMY_2D_V4 * NVAR_2D];

public:
  __device__ void
  set_x(int const tx, int const ty, int const i_var, real_t const value)
  {
    data[tx + DIM_X1 * ty + DIM_XY1 * i_var] = value;
  }

  __device__ real_t
  get_x(int const tx, int const ty, int const i_var) const
  {
    return data[tx + DIM_X1 * ty + DIM_XY1 * i_var];
  }

  __device__ void
  set_y(int const tx, int const ty, int const i_var, real_t const value)
  {
    data[tx + DIM_X2 * ty + DIM_XY2 * i_var] = value;
  }

  __device__ real_t
  get_y(int const tx, int const ty, int const i_var) const
  {
    return data[tx + DIM_X2 * ty + DIM_XY2 * i_var];
  }

}; // class QreconsShmem

/**
 * conservative (rho, rho*u, rho*v, e) to primitive variables (rho,
 * u,v,p)
 * @param[in]  u  conservative variables
 * @param[out] q  primitive    variables
 * @param[out] c  local speed of sound
 */
__DEVICE__
void
constoprim_2D(const real_t u[NVAR_2D], QShmem & q, int tx, int ty, real_t & c)
{
  const rho = FMAX(u[ID], ::gParams.smallr);

  q.set(tx, ty, ID, rho);
  q.set(tx, ty, IU, u[IU] / rho);
  q.set(tx, ty, IV, u[IV] / rho);

  // kinetic energy per mass unit
  real_t eken =
    0.5f * (q.get(tx, ty, IU) * q.get(tx, ty, IU) + q.get(tx, ty, IV) * q.get(tx, ty, IV));

  // compute pressure
  if (::gParams.cIso > 0)
  { // isothermal eos : P = c_s^2 * \rho

    q.set(tx, ty, IP, rho * (::gParams.cIso) * (::gParams.cIso));
    c = ::gParams.cIso;
  }
  else
  {

    // internal energy = total energy - kinetic energy (per mass unit)
    real_t eint = u[IP] / rho - eken;
    if (eint < 0)
    {
      PRINTF("hydro eint < 0  : e %f eken %f diff %f d %f u %f v %f\n",
             u[IP] / rho,
             eken,
             u[IP] / rho - eken,
             u[ID],
             u[IU],
             u[IV]);
    }
    // use perfect gas equation of state to compute P
    const real_t p = FMAX((::gParams.gamma0 - 1.0f) * rho * eint, rho * ::gParams.smallp);
    c = SQRT(::gParams.gamma0 * p / rho);

    q.set(tx, ty, IP, p);

  } // end compute pressure

} // constoprim_2D

/**
 * This another implementation of trace computations for 2D data;
 * slopes are computed outside; it is used when unsplitVersion = 0
 *
 * Note that :
 * - hydro slopes computations are done outside this routine
 *
 * \param[in]  q  primitive variable array (shared memory)
 * \param[in]  dtdx dt divided by dx
 * \param[in]  dtdy dt divided by dy
 * \param[in]  dir direction (either X or Y)
 * \param[out] qL the reconstructed state from the left
 * \param[out] qR the reconstructed state from the right
 */
__DEVICE__
void
trace_unsplit_hydro_2d_by_direction_shmem(QShmem const & q,
                                          int const &    tx,
                                          int const &    ty,
                                          real_t const & dtdx,
                                          real_t const & dtdy,
                                          int const      dir,
                                          QReconsShmem & qL,
                                          QReconsShmem & qR)
{

  // some aliases
  real_t & smallR = ::gParams.smallr;
  real_t & smallp = ::gParams.smallp;
  real_t & gamma = ::gParams.gamma0;

  // Cell centered values
  real_t & r = q[tx][ty][ID];
  real_t & p = q[tx][ty][IP];
  real_t & u = q[tx][ty][IU];
  real_t & v = q[tx][ty][IV];

  // Cell centered TVD slopes in X direction
  real_t drx = HALF_F * slope_scalar(q[tx][ty][ID], q[tx + 1][ty][ID], q[tx - 1][ty][ID]);
  real_t dpx = HALF_F * slope_scalar(q[tx][ty][IP], q[tx + 1][ty][IP], q[tx - 1][ty][IP]);
  real_t dux = HALF_F * slope_scalar(q[tx][ty][IU], q[tx + 1][ty][IU], q[tx - 1][ty][IU]);
  real_t dvx = HALF_F * slope_scalar(q[tx][ty][IV], q[tx + 1][ty][IV], q[tx - 1][ty][IV]);

  // Cell centered TVD slopes in Y direction
  real_t dry = HALF_F * slope_scalar(q[tx][ty][ID], q[tx][ty + 1][ID], q[tx][ty - 1][ID]);
  real_t dpy = HALF_F * slope_scalar(q[tx][ty][ID], q[tx][ty + 1][ID], q[tx][ty - 1][ID]);
  real_t duy = HALF_F * slope_scalar(q[tx][ty][ID], q[tx][ty + 1][ID], q[tx][ty - 1][ID]);
  real_t dvy = HALF_F * slope_scalar(q[tx][ty][ID], q[tx][ty + 1][ID], q[tx][ty - 1][ID]);

  // Hancock term
  const real_t sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
  const real_t su0 = (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
  const real_t sv0 = (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
  const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;

  // Update in time the  primitive variables
  r = r + sr0;
  u = u + su0;
  v = v + sv0;
  p = p + sp0;

  if (dir == IX)
  {
    const int tx2 = tx - 1;
    const int ty2 = ty - 2;

    qL[tx2][ty2][ID] = r + sr0 - drx;
    qL[tx2][ty2][IU] = u + su0 - dux;
    qL[tx2][ty2][IV] = v + sv0 - dvx;
    qL[tx2][ty2][IP] = p + sp0 - dpx;
    qL[tx2][ty2][ID] = FMAX(smallR, qL[tx2][ty2][ID]);
    qL[tx2][ty2][IP] = FMAX(smallp * qL[tx2][ty2][ID], qL[tx2][ty2][IP]);

    qR[tx2][ty2][ID] = r + sr0 + drx;
    qR[tx2][ty2][IU] = u + su0 + dux;
    qR[tx2][ty2][IV] = v + sv0 + dvx;
    qR[tx2][ty2][IP] = p + sp0 + dpx;
    qR[tx2][ty2][ID] = FMAX(smallR, qR[tx2][ty2][ID]);
    qR[tx2][ty2][IP] = FMAX(smallp * qR[tx2][ty2][ID], qR[tx2][ty2][IP]);
  }
  else if (dir == IY)
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 1;

    qL[tx2][ty2][ID] = r + sr0 - dry;
    qL[tx2][ty2][IU] = u + su0 - duy;
    qL[tx2][ty2][IV] = v + sv0 - dvy;
    qL[tx2][ty2][IP] = p + sp0 - dpy;
    qL[tx2][ty2][ID] = FMAX(smallR, qL[tx2][ty2][ID]);
    qL[tx2][ty2][IP] = FMAX(smallp * qL[tx2][ty2][ID], qL[tx2][ty2][IP]);

    qR[tx2][ty2][ID] = r + sr0 + dry;
    qR[tx2][ty2][IU] = u + su0 + duy;
    qR[tx2][ty2][IV] = v + sv0 + dvy;
    qR[tx2][ty2][IP] = p + sp0 + dpy;
    qR[tx2][ty2][ID] = FMAX(smallR, qR[tx2][ty2][ID]);
    qR[tx2][ty2][IP] = FMAX(smallp * qR[tx2][ty2][ID], qR[tx2][ty2][IP]);
  }

} // trace_unsplit_hydro_2d_by_direction_shmem
