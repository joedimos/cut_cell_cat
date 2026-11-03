#include "CPP_OPTIONS.h"

      SUBROUTINE CC_TRANSFER(state,west,south,record,ledger,
     &                       myThid)
C     Each internal face contributes one signed volume transfer and the
C     same transfer times donor concentration to both adjacent cells.
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      INTEGER myThid,i,j
      LOGICAL record
      _RL state(4,sNx,sNy),west(sNx+1,sNy),south(sNx,sNy+1)
      _RL ledger(4,sNx,sNy),tracer_flux
      _RL volume(sNx,sNy),content(sNx,sNy)
      _RL incoming(sNx,sNy),outgoing(sNx,sNy)
      _RL donor(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      volume=state(1,:,:)
      donor=0. _d 0
      WHERE (volume.GT.0. _d 0)
     & donor(1:sNx,1:sNy,1,1)=state(4,:,:)/volume
      _EXCH_XY_RL(donor,myThid)
      incoming=0. _d 0
      outgoing=0. _d 0
      content=0. _d 0
      DO j=1,sNy
       DO i=1,sNx+1
        CALL transfer(i-1,j,i,j,west(i,j),tracer_flux)
        IF (record .AND. i.LE.sNx) THEN
         ledger(1,i,j)=ledger(1,i,j)+west(i,j)
         ledger(3,i,j)=ledger(3,i,j)+tracer_flux
        ENDIF
       ENDDO
      ENDDO
      DO j=1,sNy+1
       DO i=1,sNx
        CALL transfer(i,j-1,i,j,south(i,j),tracer_flux)
        IF (record .AND. j.LE.sNy) THEN
         ledger(2,i,j)=ledger(2,i,j)+south(i,j)
         ledger(4,i,j)=ledger(4,i,j)+tracer_flux
        ENDIF
       ENDDO
      ENDDO
      IF ( ANY(outgoing.GT.volume) )
     & CALL CC_FAIL('transport exceeded donor volume',myThid)
      volume=volume-outgoing
      state(1,:,:)=volume+incoming
      WHERE (outgoing.GT.0. _d 0)
     & state(4,:,:)=volume*donor(1:sNx,1:sNy,1,1)
      state(4,:,:)=state(4,:,:)+content
      IF ( ANY(.NOT.IEEE_IS_FINITE(state)) )
     & CALL CC_FAIL('nonfinite transport state',myThid)

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
      mass=amount*donor(di,dj,1,1)
      q=SIGN(1. _d 0,f)*mass
      IF (di.GE.1.AND.di.LE.sNx.AND.dj.GE.1.AND.dj.LE.sNy)
     & outgoing(di,dj)=outgoing(di,dj)+amount
      IF (ri.GE.1.AND.ri.LE.sNx.AND.rj.GE.1.AND.rj.LE.sNy) THEN
       incoming(ri,rj)=incoming(ri,rj)+amount
       content(ri,rj)=content(ri,rj)+mass
      ENDIF
      END SUBROUTINE transfer
      END

      SUBROUTINE CC_SWE_ADVANCE(area,west,south,bed,g,interval,
     &                          state,record,ledger,myThid)
C     Unsplit hydrostatic reconstruction with local Lax-Friedrichs flux.
C     One MPI tile per rank. Seam faces use the same exchanged states
C     in the same orientation. Only the east/north tile writes the face.
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      INTEGER myThid,i,j,substeps
      LOGICAL record,atW,atE,atS,atN
      _RL area(sNx,sNy),west(sNx+1,sNy),south(sNx,sNy+1)
      _RL bed(sNx,sNy),g,interval,state(4,sNx,sNy),time_left,dt
      _RL ledger(4,sNx,sNy),speed(sNx,sNy),rate(2,sNx,sNy)
      _RL flow(1-OLx:sNx+OLx,1-OLy:sNy+OLy,4,nSx,nSy)
      _RL fx(sNx+1,sNy),fy(sNx,sNy+1)
      _RL cfl,velocity_depth_floor
      PARAMETER (cfl=0.4 _d 0, velocity_depth_floor=1. _d -10)

      atW=myXGlobalLo.EQ.1
      atE=myXGlobalLo+sNx-1.EQ.Nx
      atS=myYGlobalLo.EQ.1
      atN=myYGlobalLo+sNy-1.EQ.Ny
      time_left=interval
      substeps=0
      DO WHILE (time_left.GT.0. _d 0)
       substeps=substeps+1
       IF (substeps.GT.200000)
     &  CALL CC_FAIL('too many CFL substeps',myThid)
       flow=0. _d 0
       flow(1:sNx,1:sNy,4,1,1)=bed
       speed=0. _d 0
       rate=0. _d 0
       fx=0. _d 0
       fy=0. _d 0
       DO j=1,sNy
        DO i=1,sNx
         IF (area(i,j).GT.0. _d 0) THEN
          flow(i,j,1,1,1)=state(1,i,j)/area(i,j)
          flow(i,j,2:3,1,1)=state(2:3,i,j)
     &     /MAX(state(1,i,j),area(i,j)*velocity_depth_floor)
         ENDIF
        ENDDO
       ENDDO
C      MITgcm allocates Nr+1 exchange levels. Send two fields at once.
       CALL EXCH_3D_RL(flow(1-OLx,1-OLy,1,1,1),2,myThid)
       CALL EXCH_3D_RL(flow(1-OLx,1-OLy,3,1,1),2,myThid)
       DO j=1,sNy
        DO i=1,sNx+1
         IF ( (i.EQ.1.AND.atW) .OR. (i.EQ.sNx+1.AND.atE) ) CYCLE
         IF (west(i,j).GT.0. _d 0)
     &    CALL face(i-1,j,i,j,1,west(i,j),fx(i,j))
        ENDDO
       ENDDO
       DO j=1,sNy+1
        IF ( (j.EQ.1.AND.atS) .OR. (j.EQ.sNy+1.AND.atN) ) CYCLE
        DO i=1,sNx
         IF (south(i,j).GT.0. _d 0)
     &    CALL face(i,j-1,i,j,2,south(i,j),fy(i,j))
        ENDDO
       ENDDO
       dt=time_left
       DO j=1,sNy
        DO i=1,sNx
         IF (area(i,j).GT.0. _d 0) THEN
C         Recover the immersed normal times length by polygon closure.
C         Physical box faces reflect. MPI seams pass shared transport.
          CALL wall(i,j,west(i,j)-west(i+1,j),
     &                  south(i,j)-south(i,j+1))
          IF (i.EQ.1.AND.atW) CALL wall(i,j,-west(i,j),0. _d 0)
          IF (i.EQ.sNx.AND.atE) CALL wall(i,j,west(i+1,j),0. _d 0)
          IF (j.EQ.1.AND.atS) CALL wall(i,j,0. _d 0,-south(i,j))
          IF (j.EQ.sNy.AND.atN) CALL wall(i,j,0. _d 0,south(i,j+1))
          IF (speed(i,j).GT.0. _d 0)
     &     dt=MIN(dt,cfl*area(i,j)/speed(i,j))
         ENDIF
        ENDDO
       ENDDO
       IF (.NOT.IEEE_IS_FINITE(dt) .OR. dt.LE.0. _d 0)
     &  CALL CC_FAIL('invalid local CFL step',myThid)
       dt=-dt
       _GLOBAL_MAX_RL(dt,myThid)
       dt=-dt
       IF (time_left-dt.EQ.time_left)
     &  CALL CC_FAIL('unrepresentable global CFL step',myThid)
       CALL CC_TRANSFER(state,dt*fx,dt*fy,record,ledger,myThid)
       state(2:3,:,:)=state(2:3,:,:)+dt*rate
       IF (ANY(.NOT.IEEE_IS_FINITE(state)))
     &  CALL CC_FAIL('nonfinite shallow-water state',myThid)
       time_left=time_left-dt
      ENDDO

      CONTAINS

      SUBROUTINE face(il,jl,ir,jr,axis,length,mass_rate)
      INTEGER il,jl,ir,jr,axis
      _RL length,mass_rate,z,hl,hr,ul,ur,a,left,right,pressure
      _RL bl,br,dl,dr,vl(2),vr(2)
      _RL advective(2)
      bl=flow(il,jl,4,1,1)
      br=flow(ir,jr,4,1,1)
      dl=flow(il,jl,1,1,1)
      dr=flow(ir,jr,1,1,1)
      vl=flow(il,jl,2:3,1,1)
      vr=flow(ir,jr,2:3,1,1)
      z=MAX(bl,br)
      hl=MAX(0. _d 0,MIN(dl,dl+(bl-z)))
      hr=MAX(0. _d 0,MIN(dr,dr+(br-z)))
      ul=vl(axis)
      ur=vr(axis)
      a=MAX(ABS(ul)+SQRT(g*hl),ABS(ur)+SQRT(g*hr))
      left=0.5 _d 0*hl*(ul+a)
      right=0.5 _d 0*hr*(ur-a)
      mass_rate=length*(left+right)
      advective=length*(left*vl+right*vr)
C     Hydrostatic correction plus the baseline wall pressure leaves
C     F_mom - g/2*h_star^2*n in each cell's residual. Evaluate the
C     pressure difference without subtracting two large pressures.
      pressure=0.25 _d 0*g*length*(hr-hl)*(hr+hl)
      IF (il.GE.1.AND.jl.GE.1) THEN
       rate(:,il,jl)=rate(:,il,jl)-advective
       rate(axis,il,jl)=rate(axis,il,jl)-pressure
       speed(il,jl)=speed(il,jl)+length*a
      ENDIF
      IF (ir.LE.sNx.AND.jr.LE.sNy) THEN
       rate(:,ir,jr)=rate(:,ir,jr)+advective
       rate(axis,ir,jr)=rate(axis,ir,jr)-pressure
       speed(ir,jr)=speed(ir,jr)+length*a
      ENDIF
      END SUBROUTINE face

      SUBROUTINE wall(i,j,nx_len,ny_len)
      INTEGER i,j
      _RL nx_len,ny_len,length,un,a,extra,h
      length=HYPOT(nx_len,ny_len)
      IF (length.EQ.0. _d 0) RETURN
      un=(flow(i,j,2,1,1)*nx_len+flow(i,j,3,1,1)*ny_len)
     &   /length
      h=flow(i,j,1,1,1)
      a=ABS(un)+SQRT(g*h)
C     Slip-reflected ghost, q_g = q - 2 (q.n)n. Its Rusanov mass flux
C     vanishes exactly. The excess normal force is h*un*(un+a).
      extra=h*un*(un+a)
      rate(1,i,j)=rate(1,i,j)-extra*nx_len
      rate(2,i,j)=rate(2,i,j)-extra*ny_len
      speed(i,j)=speed(i,j)+length*a
      END SUBROUTINE wall
      END
