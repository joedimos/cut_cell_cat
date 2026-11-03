#include "CPP_OPTIONS.h"

      SUBROUTINE CC_TRANSFER(nx,ny,state,west,south,record,ledger)
C     Each internal face contributes one signed volume transfer and the
C     same transfer times donor concentration to both adjacent cells.
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER nx,ny,i,j
      LOGICAL record
      _RL state(4,nx,ny),west(nx+1,ny),south(nx,ny+1)
      _RL ledger(4,nx,ny),tracer_flux
      _RL volume(nx,ny),content(nx,ny),donor(nx,ny)
      _RL incoming(nx,ny),outgoing(nx,ny)
      volume=state(1,:,:)
      donor=0. _d 0
      WHERE (volume.GT.0. _d 0) donor=state(4,:,:)/volume
      incoming=0. _d 0
      outgoing=0. _d 0
      content=0. _d 0
      DO j=1,ny
       DO i=2,nx
        CALL transfer(i-1,j,i,j,west(i,j),tracer_flux)
        IF (record) THEN
         ledger(1,i,j)=ledger(1,i,j)+west(i,j)
         ledger(3,i,j)=ledger(3,i,j)+tracer_flux
        ENDIF
       ENDDO
      ENDDO
      DO j=2,ny
       DO i=1,nx
        CALL transfer(i,j-1,i,j,south(i,j),tracer_flux)
        IF (record) THEN
         ledger(2,i,j)=ledger(2,i,j)+south(i,j)
         ledger(4,i,j)=ledger(4,i,j)+tracer_flux
        ENDIF
       ENDDO
      ENDDO
      IF ( ANY(outgoing.GT.volume) )
     & STOP 'CUTCELL transport exceeded available donor volume'
      volume=volume-outgoing
      state(1,:,:)=volume+incoming
      WHERE (outgoing.GT.0. _d 0) state(4,:,:)=volume*donor
      state(4,:,:)=state(4,:,:)+content
      IF ( ANY(.NOT.IEEE_IS_FINITE(state)) )
     & STOP 'CUTCELL nonfinite transport state'

      CONTAINS

      SUBROUTINE transfer(il,jl,ir,jr,f,q)
      INTEGER il,jl,ir,jr,di,dj,ri,rj
      _RL f,q,amount,mass
      q=0. _d 0
      IF (f.EQ.0. _d 0) RETURN
      di=il
      dj=jl
      ri=ir
      rj=jr
      IF (f.LT.0. _d 0) THEN
       di=ir
       dj=jr
       ri=il
       rj=jl
      ENDIF
      amount=ABS(f)
      mass=amount*donor(di,dj)
      q=SIGN(1. _d 0,f)*mass
      outgoing(di,dj)=outgoing(di,dj)+amount
      incoming(ri,rj)=incoming(ri,rj)+amount
      content(ri,rj)=content(ri,rj)+mass
      END SUBROUTINE transfer
      END

      SUBROUTINE CC_SWE_ADVANCE(nx,ny,area,west,south,bed,g,interval,
     &                          state,record,ledger)
C     Unsplit hydrostatic reconstruction with local Lax-Friedrichs flux.
C     Piecewise constant bed/state. Cartesian grid clipped by straight
C     slip walls, at most one immersed wall segment per cell.
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER nx,ny,i,j,substeps
      LOGICAL record
      _RL area(nx,ny),west(nx+1,ny),south(nx,ny+1),bed(nx,ny),g
      _RL interval,state(4,nx,ny),time_left,dt
      _RL ledger(4,nx,ny)
      _RL depth(nx,ny),vel(2,nx,ny),speed(nx,ny),rate(2,nx,ny)
      _RL fx(nx+1,ny),fy(nx,ny+1)
      _RL cfl,velocity_depth_floor
      PARAMETER (cfl=0.4 _d 0, velocity_depth_floor=1. _d -10)

      time_left=interval
      substeps=0
      DO WHILE (time_left.GT.0. _d 0)
       substeps=substeps+1
       IF (substeps.GT.200000) STOP 'CUTCELL too many CFL substeps'
       depth=0. _d 0
       vel=0. _d 0
       speed=0. _d 0
       rate=0. _d 0
       fx=0. _d 0
       fy=0. _d 0
       DO j=1,ny
        DO i=1,nx
         IF (area(i,j).GT.0. _d 0) THEN
          depth(i,j)=state(1,i,j)/area(i,j)
          vel(:,i,j)=state(2:3,i,j)
     &     /MAX(state(1,i,j),area(i,j)*velocity_depth_floor)
         ENDIF
        ENDDO
       ENDDO
       DO j=1,ny
        DO i=2,nx
         IF (west(i,j).GT.0. _d 0)
     &    CALL face(i-1,j,i,j,1,west(i,j),fx(i,j))
        ENDDO
       ENDDO
       DO j=2,ny
        DO i=1,nx
         IF (south(i,j).GT.0. _d 0)
     &    CALL face(i,j-1,i,j,2,south(i,j),fy(i,j))
        ENDDO
       ENDDO
       dt=time_left
       DO j=1,ny
        DO i=1,nx
         IF (area(i,j).GT.0. _d 0) THEN
C         The immersed normal times length is recovered by closure of
C         the clipped polygon. Treat box faces as separate slip walls.
          CALL wall(i,j,west(i,j)-west(i+1,j),
     &                  south(i,j)-south(i,j+1))
          IF (i.EQ.1) CALL wall(i,j,-west(i,j),0. _d 0)
          IF (i.EQ.nx) CALL wall(i,j,west(i+1,j),0. _d 0)
          IF (j.EQ.1) CALL wall(i,j,0. _d 0,-south(i,j))
          IF (j.EQ.ny) CALL wall(i,j,0. _d 0,south(i,j+1))
          IF (speed(i,j).GT.0. _d 0)
     &     dt=MIN(dt,cfl*area(i,j)/speed(i,j))
         ENDIF
        ENDDO
       ENDDO
       IF (.NOT.IEEE_IS_FINITE(dt) .OR. dt.LE.0. _d 0
     &     .OR. time_left-dt.EQ.time_left)
     &  STOP 'CUTCELL invalid CFL substep'
       CALL CC_TRANSFER(nx,ny,state,dt*fx,dt*fy,record,ledger)
       state(2:3,:,:)=state(2:3,:,:)+dt*rate
       IF (ANY(.NOT.IEEE_IS_FINITE(state)))
     &  STOP 'CUTCELL nonfinite shallow-water state'
       time_left=time_left-dt
      ENDDO

      CONTAINS

      SUBROUTINE face(il,jl,ir,jr,axis,length,mass_rate)
      INTEGER il,jl,ir,jr,axis
      _RL length,mass_rate,z,hl,hr,ul,ur,a,left,right,pressure
      _RL advective(2)
      z=MAX(bed(il,jl),bed(ir,jr))
      hl=MAX(0. _d 0,MIN(depth(il,jl),depth(il,jl)+(bed(il,jl)-z)))
      hr=MAX(0. _d 0,MIN(depth(ir,jr),depth(ir,jr)+(bed(ir,jr)-z)))
      ul=vel(axis,il,jl)
      ur=vel(axis,ir,jr)
      a=MAX(ABS(ul)+SQRT(g*hl),ABS(ur)+SQRT(g*hr))
      left=0.5 _d 0*hl*(ul+a)
      right=0.5 _d 0*hr*(ur-a)
      mass_rate=length*(left+right)
      advective=length*(left*vel(:,il,jl)+right*vel(:,ir,jr))
C     Hydrostatic correction plus the baseline wall pressure leaves
C     F_mom - g/2*h_star^2*n in each cell's residual. Evaluate the
C     pressure difference without subtracting two large pressures.
      pressure=0.25 _d 0*g*length*(hr-hl)*(hr+hl)
      rate(:,il,jl)=rate(:,il,jl)-advective
      rate(:,ir,jr)=rate(:,ir,jr)+advective
      rate(axis,il,jl)=rate(axis,il,jl)-pressure
      rate(axis,ir,jr)=rate(axis,ir,jr)-pressure
      speed(il,jl)=speed(il,jl)+length*a
      speed(ir,jr)=speed(ir,jr)+length*a
      END SUBROUTINE face

      SUBROUTINE wall(i,j,nx_len,ny_len)
      INTEGER i,j
      _RL nx_len,ny_len,length,un,a,extra
      length=HYPOT(nx_len,ny_len)
      IF (length.EQ.0. _d 0) RETURN
      un=(vel(1,i,j)*nx_len+vel(2,i,j)*ny_len)/length
      a=ABS(un)+SQRT(g*depth(i,j))
C     Slip-reflected ghost, q_g = q - 2 (q.n)n. Its Rusanov mass flux
C     vanishes exactly. The excess normal force is h*un*(un+a).
      extra=depth(i,j)*un*(un+a)
      rate(1,i,j)=rate(1,i,j)-extra*nx_len
      rate(2,i,j)=rate(2,i,j)-extra*ny_len
      speed(i,j)=speed(i,j)+length*a
      END SUBROUTINE wall
      END
