#include "real_type.h"
#include "constants.h"
#include "utils/config/ConfigMap.h"
#include <cuda/std/algorithm> // for cuda::std::max

// #ifndef MAX
// #  define MAX(a, b) ((a) > (b) ? (a) : (b))
// #endif

/**
 * Primitive variables array (for shared memory storage).
 *
 * Example 8x8 ghosted block of cells:
 * - 2 ghosts cells all arround
 * - the inner core is 4x4
 *    _______________
 *   |_|_|_|_|_|_|_|_|
 *   |_|_|_|_|_|_|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|_|_|_|_|_|_|
 *   |_|_|_|_|_|_|_|_|
 */
class QPrim
{
public:
  // static constexpr int    DIM_XY = UNSPLIT_BLOCK_DIMX_2D_V3 * UNSPLIT_BLOCK_DIMY_2D_V3;
  // static constexpr size_t NUM_ELEMENTS = static_cast<size_t>(DIM_XY * NVAR_2D);

  __host__ __device__ static size_t
  required_size_in_bytes(int dim_x, int dim_y)
  {
    return static_cast<size_t>(dim_x * dim_y * NVAR_2D) * sizeof(real_t);
  }

private:
  //! externally managed memory (e.g. cuda shared memory)
  real_t * m_data;

  //! size along x axis
  int m_dim_x;

  //! size along y axis
  int m_dim_y;

  //! total x-y size
  int m_dim_xy;

public:
  __device__
  QPrim()
  {}

  __device__
  QPrim(void * ptr, int dim_x, int dim_y)
    : m_data(reinterpret_cast<real_t *>(ptr))
    , m_dim_x(dim_x)
    , m_dim_y(dim_y)
    , m_dim_xy(dim_x * dim_y)
  {}

  __device__ real_t &
  operator()(int const tx, int const ty, int const i_var)
  {
    return m_data[tx + m_dim_x * ty + m_dim_xy * i_var];
  }

  __device__ real_t const &
  operator()(int const tx, int const ty, int const i_var) const
  {
    return m_data[tx + m_dim_x * ty + m_dim_xy * i_var];
  }
}; // class QPrim

/**
 * Reconstructed states array (for shared memory storage).
 *
 * When doing computing along X axis, we need to reconstruct states on the left and right faces of
 * all cells marked with symbol "x" and "o". Ghost cells (void) are necessary for computing slopes
 *    _______________
 *   |_|_|_|_|_|_|_|_|
 *   |_|_|_|_|_|_|_|_|
 *   |_|o|x|x|x|x|o|_|
 *   |_|o|x|x|x|x|o|_|
 *   |_|o|x|x|x|x|o|_|
 *   |_|o|x|x|x|x|o|_|
 *   |_|_|_|_|_|_|_|_|
 *   |_|_|_|_|_|_|_|_|
 *
 * When doing computing along Y axis, we need to reconstruct states on the left and right faces of
 * all cells marked with symbol "x" and "o". Ghost cells (void) are necessary for computing slopes
 *    _______________
 *   |_|_|_|_|_|_|_|_|
 *   |_|_|o|o|o|o|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|x|x|x|x|_|_|
 *   |_|_|o|o|o|o|_|_|
 *   |_|_|_|_|_|_|_|_|
 *
 *
 * The shared memory array has 2 memory layouts depending whether doing computations along X or Y
 * axis
 */
class QRecons
{
public:
  __host__ __device__ static size_t
  required_size_in_bytes(int dim_x, int dim_y)
  {
    const auto dim_xy1 = static_cast<size_t>((dim_x - 2) * (dim_y - 4));
    const auto dim_xy2 = static_cast<size_t>((dim_x - 4) * (dim_y - 2));
    return cuda::std::max(dim_xy1, dim_xy2) * NVAR_2D * sizeof(real_t);
  }

private:
  //! externally managed memory (e.g. cuda shared memory)
  real_t * m_data;

  //! size along x axis when computing along x axis
  int m_dim_x1;

  //! size along y axis when computing along x axis
  int m_dim_y1;

  //! total number of cells excluding ghost cells when computing along x axis
  int m_dim_xy1;

  //! size along x axis when computing along y axis
  int m_dim_x2;

  //! size along y axis when computing along y axis
  int m_dim_y2;

  //! total number of cells excluding ghost cells when computing along y axis
  int m_dim_xy2;

public:
  __device__
  QRecons()
  {}

  __device__
  QRecons(void * ptr, int dim_x, int dim_y)
    : m_data(reinterpret_cast<real_t *>(ptr))
    , m_dim_x1(dim_x - 2)
    , m_dim_y1(dim_y - 4)
    , m_dim_xy1(m_dim_x1 * m_dim_y1)
    , m_dim_x2(dim_x - 4)
    , m_dim_y2(dim_y - 2)
    , m_dim_xy2(m_dim_x2 * m_dim_y2)
  {}

  __device__ real_t &
  operator()(int const tx, int const ty, int const i_var, const int dir)
  {
    if (dir == IX)
      return m_data[tx + m_dim_x1 * ty + m_dim_xy1 * i_var];
    else if (dir == IY)
      return m_data[tx + m_dim_x2 * ty + m_dim_xy2 * i_var];
    return m_data[0];
  }

  __device__ real_t const &
  operator()(int const tx, int const ty, int const i_var, const int dir) const
  {
    if (dir == IX)
      return m_data[tx + m_dim_x1 * ty + m_dim_xy1 * i_var];
    else if (dir == IY)
      return m_data[tx + m_dim_x2 * ty + m_dim_xy2 * i_var];
    return m_data[0];
  }

}; // class QRecons

/**
 * \brief Gathering configuration information for running CUDA kernel kernel_godunov_unsplit_2d_v3.
 *
 * Example : amount of required shared memory
 */
class GodunovUnsplitConfigV3
{
public:
  __device__ static size_t
  offset_bytes_qm(int dim_x, int dim_y)
  {
    return QPrim::required_size_in_bytes(dim_x, dim_y);
  };

  __device__ static size_t
  offset_bytes_qp(int dim_x, int dim_y)
  {
    return offset_bytes_qm(dim_x, dim_y) + QRecons::required_size_in_bytes(dim_x, dim_y);
  }

  __device__ static size_t
  offset_bytes_flux(int dim_x, int dim_y)
  {
    return offset_bytes_qp(dim_y, dim_y) + QRecons::required_size_in_bytes(dim_x, dim_y);
  }

  static size_t
  required_shared_mem_bytes(int dim_x, int dim_y)
  {
    return QPrim::required_size_in_bytes(dim_x, dim_y) +
           3 * QRecons::required_size_in_bytes(dim_x, dim_y);
  }

  static size_t
  required_shared_mem_bytes(ConfigMap const & config_map)
  {
    const auto dim_x = config_map.getInteger("hydro_unsplit_version_3", "dim_x", 16);
    const auto dim_y = config_map.getInteger("hydro_unsplit_version_3", "dim_y", 16);

    return required_shared_mem_bytes(dim_x, dim_y);
  }
};

/**
 * conservative (rho, rho*u, rho*v, e) to primitive variables (rho,
 * u,v,p)
 * \param[in]  u  conservative variables state in a cell cell
 * \param[out] q  array of primitive variable states (ghosted block)
 * \param[out] c  local speed of sound
 */
__DEVICE__
void
constoprim_2D(const real_t u[NVAR_2D], QPrim & q, int tx, int ty, real_t & c)
{
  const real_t rho = FMAX(u[ID], ::gParams.smallr);

  q(tx, ty, ID) = rho;
  q(tx, ty, IU) = u[IU] / rho;
  q(tx, ty, IV) = u[IV] / rho;

  // kinetic energy per mass unit
  real_t eken = 0.5f * (q(tx, ty, IU) * q(tx, ty, IU) + q(tx, ty, IV) * q(tx, ty, IV));

  // compute pressure
  if (::gParams.cIso > 0)
  { // isothermal eos : P = c_s^2 * \rho

    q(tx, ty, IP) = rho * (::gParams.cIso) * (::gParams.cIso);
    c = ::gParams.cIso;
  }
  else
  {

    // internal energy = total energy - kinetic energy (per mass unit)
    real_t eint = u[IP] / rho - eken;
    if (eint < 0)
    {
      PRINTF("hydro eint < 0  : e %f eken %f diff %f d %f u %f v %f - tx=%d ty=%d\n",
             u[IP] / rho,
             eken,
             u[IP] / rho - eken,
             u[ID],
             u[IU],
             u[IV],
             tx,
             ty);
    }
    // use perfect gas equation of state to compute P
    const real_t p = FMAX((::gParams.gamma0 - 1.0f) * rho * eint, rho * ::gParams.smallp);
    c = SQRT(::gParams.gamma0 * p / rho);

    q(tx, ty, IP) = p;

  } // end compute pressure

} // constoprim_2D

/**
 * This another implementation of trace computations for 2D data;
 * slopes are computed outside; it is used when unsplitVersion = 0
 *
 * Note that :
 * - hydro slopes computations are done outside this routine
 *
 * \param[in]  q  array of primitive variable in shared memory (ghosted block)
 * \param[in] tx x coordinate identifying a cell in the ghosted block
 * \param[in] ty y coordinate identifying a cell in the ghosted block
 * \param[in]  dtdx dt divided by dx
 * \param[in]  dtdy dt divided by dy
 * \param[in]  dir direction (either IX or IY)
 * \param[out] qm array of reconstructed states (left face of a cell)
 * \param[out] qp array of reconstructed state  (right face of a cell)
 */
__DEVICE__
void
trace_unsplit_hydro_2d_by_direction_shmem(QPrim const &  q,
                                          int const &    tx,
                                          int const &    ty,
                                          real_t const & dtdx,
                                          real_t const & dtdy,
                                          int const      dir,
                                          QRecons &      qm,
                                          QRecons &      qp)
{

  // some aliases
  real_t & smallR = ::gParams.smallr;
  real_t & smallp = ::gParams.smallp;
  real_t & gamma = ::gParams.gamma0;

  // Cell centered values
  real_t r = q(tx, ty, ID);
  real_t p = q(tx, ty, IP);
  real_t u = q(tx, ty, IU);
  real_t v = q(tx, ty, IV);

  // Cell centered TVD slopes in X direction
  real_t drx = slope_scalar(q(tx, ty, ID), q(tx + 1, ty, ID), q(tx - 1, ty, ID));
  real_t dpx = slope_scalar(q(tx, ty, IP), q(tx + 1, ty, IP), q(tx - 1, ty, IP));
  real_t dux = slope_scalar(q(tx, ty, IU), q(tx + 1, ty, IU), q(tx - 1, ty, IU));
  real_t dvx = slope_scalar(q(tx, ty, IV), q(tx + 1, ty, IV), q(tx - 1, ty, IV));

  // Cell centered TVD slopes in Y direction
  real_t dry = slope_scalar(q(tx, ty, ID), q(tx, ty + 1, ID), q(tx, ty - 1, ID));
  real_t dpy = slope_scalar(q(tx, ty, IP), q(tx, ty + 1, IP), q(tx, ty - 1, IP));
  real_t duy = slope_scalar(q(tx, ty, IU), q(tx, ty + 1, IU), q(tx, ty - 1, IU));
  real_t dvy = slope_scalar(q(tx, ty, IV), q(tx, ty + 1, IV), q(tx, ty - 1, IV));

  // Hancock term
  const real_t sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
  const real_t su0 = (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
  const real_t sv0 = (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
  const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;

  // Update in time the  primitive variables
  r = r + HALF_F * sr0;
  u = u + HALF_F * su0;
  v = v + HALF_F * sv0;
  p = p + HALF_F * sp0;

  if (dir == IX)
  {
    const int tx2 = tx - 1;
    const int ty2 = ty - 2;

    qm(tx2, ty2, ID, IX) = r - HALF_F * drx;
    qm(tx2, ty2, IU, IX) = u - HALF_F * dux;
    qm(tx2, ty2, IV, IX) = v - HALF_F * dvx;
    qm(tx2, ty2, IP, IX) = p - HALF_F * dpx;
    qm(tx2, ty2, ID, IX) = FMAX(smallR, qm(tx2, ty2, ID, IX));
    qm(tx2, ty2, IP, IX) = FMAX(smallp * qm(tx2, ty2, ID, IX), qm(tx2, ty2, IP, IX));

    qp(tx2, ty2, ID, IX) = r + HALF_F * drx;
    qp(tx2, ty2, IU, IX) = u + HALF_F * dux;
    qp(tx2, ty2, IV, IX) = v + HALF_F * dvx;
    qp(tx2, ty2, IP, IX) = p + HALF_F * dpx;
    qp(tx2, ty2, ID, IX) = FMAX(smallR, qp(tx2, ty2, ID, IX));
    qp(tx2, ty2, IP, IX) = FMAX(smallp * qp(tx2, ty2, ID, IX), qp(tx2, ty2, IP, IX));
  }
  else if (dir == IY)
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 1;

    qm(tx2, ty2, ID, IY) = r - HALF_F * dry;
    qm(tx2, ty2, IU, IY) = u - HALF_F * duy;
    qm(tx2, ty2, IV, IY) = v - HALF_F * dvy;
    qm(tx2, ty2, IP, IY) = p - HALF_F * dpy;
    qm(tx2, ty2, ID, IY) = FMAX(smallR, qm(tx2, ty2, ID, IY));
    qm(tx2, ty2, IP, IY) = FMAX(smallp * qm(tx2, ty2, ID, IY), qm(tx2, ty2, IP, IY));

    qp(tx2, ty2, ID, IY) = r + HALF_F * dry;
    qp(tx2, ty2, IU, IY) = u + HALF_F * duy;
    qp(tx2, ty2, IV, IY) = v + HALF_F * dvy;
    qp(tx2, ty2, IP, IY) = p + HALF_F * dpy;
    qp(tx2, ty2, ID, IY) = FMAX(smallR, qp(tx2, ty2, ID, IY));
    qp(tx2, ty2, IP, IY) = FMAX(smallp * qp(tx2, ty2, ID, IY), qp(tx2, ty2, IP, IY));
  }

} // trace_unsplit_hydro_2d_by_direction_shmem

/**
 * Riemann solver, equivalent to riemann_hllc in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * Hydro ONLY.
 *
 * \param[in] qm array of reconstructed states on the cell left face
 * \param[in] qp array of reconstructed states on the cell right face
 * \param[in] tx2 x coordinate identifying a cell in the extended inner core
 * \param[in] ty2 y coordinate identifying a cell in the extended inner core
 * \param[out] flux  array of Riemann fluxes (one per face)
 */
template <size_t dim, int dir>
__DEVICE__ void
riemann_hllc(QRecons const & qm, QRecons const & qp, int tx2, int ty2, QRecons & flux)
{

  // offset to access qp
  int offset_x = 0;
  int offset_y = 0;

  if constexpr (dir == IX)
    offset_x = -1;
  if constexpr (dir == IY)
    offset_y = -1;

  // swap velocity indexes
  int IUU = IU;
  int IVV = IV;
  int IWW = IW;
  if constexpr (dir == IY)
  {
    IUU = IV;
    IVV = IU;
  }
  else if constexpr (dir == IZ)
  {
    IUU = IW;
    IWW = IU;
  }

  const real_t entho = ONE_F / (gParams.gamma0 - ONE_F);

  // Left variables
  const real_t rl = FMAX(qp(tx2 + offset_x, ty2 + offset_y, ID, dir), gParams.smallr);
  const real_t pl = FMAX(qp(tx2 + offset_x, ty2 + offset_y, IP, dir), rl * gParams.smallp);
  const real_t ul = qp(tx2 + offset_x, ty2 + offset_y, IUU, dir);

  real_t ecinl = HALF_F * rl * ul * ul;
  ecinl += HALF_F * rl * qp(tx2 + offset_x, ty2 + offset_y, IVV, dir) *
           qp(tx2 + offset_x, ty2 + offset_y, IVV, dir);
  if constexpr (dim == 3)
    ecinl += HALF_F * rl * qp(tx2 + offset_x, ty2 + offset_y, IWW, dir) *
             qp(tx2 + offset_x, ty2 + offset_y, IWW, dir);

  const real_t   etotl = pl * entho + ecinl;
  real_t const & ptotl = pl;

  // Right variables
  const real_t rr = FMAX(qm(tx2, ty2, ID, dir), gParams.smallr);
  const real_t pr = FMAX(qm(tx2, ty2, IP, dir), rr * gParams.smallp);
  const real_t ur = qm(tx2, ty2, IUU, dir);

  real_t ecinr = HALF_F * rr * ur * ur;
  ecinr += HALF_F * rr * qm(tx2, ty2, IVV, dir) * qm(tx2, ty2, IVV, dir);
  if constexpr (dim == 3)
    ecinr += HALF_F * rr * qm(tx2, ty2, IWW, dir) * qm(tx2, ty2, IWW, dir);

  const real_t   etotr = pr * entho + ecinr;
  real_t const & ptotr = pr;

  // Find the largest eigenvalues in the normal direction to the interface
  const real_t cfastl = SQRT(FMAX(gParams.gamma0 * pl / rl, gParams.smallc * gParams.smallc));
  const real_t cfastr = SQRT(FMAX(gParams.gamma0 * pr / rr, gParams.smallc * gParams.smallc));

  // Compute HLL wave speed
  const real_t SL = FMIN(ul, ur) - FMAX(cfastl, cfastr);
  const real_t SR = FMAX(ul, ur) + FMAX(cfastl, cfastr);

  // Compute lagrangian sound speed
  const real_t rcl = rl * (ul - SL);
  const real_t rcr = rr * (SR - ur);

  // Compute acoustic star state
  const real_t ustar = (rcr * ur + rcl * ul + (ptotl - ptotr)) / (rcr + rcl);
  const real_t ptotstar = (rcr * ptotl + rcl * ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);

  // Left star region variables
  const real_t rstarl = rl * (SL - ul) / (SL - ustar);
  const real_t etotstarl = ((SL - ul) * etotl - ptotl * ul + ptotstar * ustar) / (SL - ustar);

  // Right star region variables
  const real_t rstarr = rr * (SR - ur) / (SR - ustar);
  const real_t etotstarr = ((SR - ur) * etotr - ptotr * ur + ptotstar * ustar) / (SR - ustar);

  // Sample the solution at x/t=0
  real_t ro, uo, ptoto, etoto;
  if (SL > ZERO_F)
  {
    ro = rl;
    uo = ul;
    ptoto = ptotl;
    etoto = etotl;
  }
  else if (ustar > ZERO_F)
  {
    ro = rstarl;
    uo = ustar;
    ptoto = ptotstar;
    etoto = etotstarl;
  }
  else if (SR > ZERO_F)
  {
    ro = rstarr;
    uo = ustar;
    ptoto = ptotstar;
    etoto = etotstarr;
  }
  else
  {
    ro = rr;
    uo = ur;
    ptoto = ptotr;
    etoto = etotr;
  }

  // Compute the Godunov flux
  flux(tx2, ty2, ID, dir) = ro * uo;
  flux(tx2, ty2, IUU, dir) = ro * uo * uo + ptoto;
  flux(tx2, ty2, IP, dir) = (etoto + ptoto) * uo;
  if (flux(tx2, ty2, ID, dir) > ZERO_F)
  {
    flux(tx2, ty2, IVV, dir) =
      flux(tx2, ty2, ID, dir) * qp(tx2 + offset_x, ty2 + offset_y, IVV, dir);
  }
  else
  {
    flux(tx2, ty2, IVV, dir) = flux(tx2, ty2, ID, dir) * qm(tx2, ty2, IVV, dir);
  }

  if constexpr (dim == 3)
  {
    if (flux(tx2, ty2, ID, dir) > ZERO_F)
    {
      flux(tx2, ty2, IWW, dir) =
        flux(tx2, ty2, ID, dir) * qp(tx2 + offset_x, ty2 + offset_y, IWW, dir);
    }
    else
    {
      flux(tx2, ty2, IWW, dir) = flux(tx2, ty2, ID, dir) * qm(tx2, ty2, IWW, dir);
    }
  }

} // riemann_hllc_2d

/**
 * Wrapper routine to call the right Riemann solver routine according
 * to the global parameter riemannSolver
 *
 * \param[in] qm array of reconstructed states on the cell left face
 * \param[in] qp array of reconstructed states on the cell right face
 * \param[in] tx2 x coordinate identifying a cell in the extended inner core
 * \param[in] ty2 y coordinate identifying a cell in the extended inner core
 * \param[out] flux  array of Riemann fluxes (one per face)
 *
 * template parameter:
 * @tparam NVAR size of state vector (can only be NVAR_2D, NVAR_3D or NVAR_MHD)
 */
template <size_t dim, int dir>
__DEVICE__ void
riemann(QRecons const & qm, QRecons const & qp, int tx2, int ty2, QRecons & flux)
{
  if (gParams.riemannSolver == HLLC)
  {
    riemann_hllc<dim, dir>(qm, qp, tx2, ty2, flux);
  }

} // riemann
