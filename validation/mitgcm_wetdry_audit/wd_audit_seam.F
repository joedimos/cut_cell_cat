#include "WETDRY_OPTIONS.h"

C     Read the transports consumed by the two sides of each tile seam.
C     Global face indices use MITgcm's west/south C-grid convention.
      SUBROUTINE WD_AUDIT_SEAM( bi, bj, k, uTrans, vTrans,
     I                         myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "WETDRY.h"
      INTEGER bi, bj, k, myIter, myThid
      _RL uTrans(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vTrans(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER i, j, gx, gy, tx, ty

      IF ( .NOT.usingCartesianGrid .OR. .NOT.usingZCoords
     &     .OR. .NOT.fluidIsWater .OR. deepAtmosphere
     &     .OR. useOBCS .OR. nThreads.NE.1
     &     .OR. deepFacC(k).NE.oneRL .OR. rhoFacC(k).NE.oneRL
     &     .OR. implicDiv2Dflow.NE.oneRL ) THEN
        STOP 'WD_AUDIT_SEAM: unsupported configuration'
      ENDIF

      gx = myXGlobalLo - 1 + (bi-1)*sNx
      gy = myYGlobalLo - 1 + (bj-1)*sNy
      tx = gx/sNx + 1
      ty = gy/sNy + 1
      DO j=1,sNy
       IF ( gx.GT.0 ) CALL WD_AUDIT_RECORD(
     I   myIter, k, 1, gx+1, gy+j, -1, tx, ty,
     I   wdMaskW(1,j,bi,bj), maskC(1,j,k,bi,bj),
     I   -deltaTFreeSurf*uTrans(1,j), myThid )
       IF ( gx+sNx.LT.Nx ) CALL WD_AUDIT_RECORD(
     I   myIter, k, 1, gx+sNx+1, gy+j, 1, tx, ty,
     I   wdMaskW(sNx+1,j,bi,bj), maskC(sNx,j,k,bi,bj),
     I   deltaTFreeSurf*uTrans(sNx+1,j), myThid )
      ENDDO
      DO i=1,sNx
       IF ( gy.GT.0 ) CALL WD_AUDIT_RECORD(
     I   myIter, k, 2, gx+i, gy+1, -1, tx, ty,
     I   wdMaskS(i,1,bi,bj), maskC(i,1,k,bi,bj),
     I   -deltaTFreeSurf*vTrans(i,1), myThid )
       IF ( gy+sNy.LT.Ny ) CALL WD_AUDIT_RECORD(
     I   myIter, k, 2, gx+i, gy+sNy+1, 1, tx, ty,
     I   wdMaskS(i,sNy+1,bi,bj), maskC(i,sNy,k,bi,bj),
     I   deltaTFreeSurf*vTrans(i,sNy+1), myThid )
      ENDDO
      RETURN
      END

      SUBROUTINE WD_AUDIT_RECORD( it, k, axis, i, j, side, tx, ty,
     I                           gate, active, volume, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      INTEGER it, k, axis, i, j, side, tx, ty, myThid
      INTEGER igate, iactive
      _RS gate, active
      _RL volume
      CHARACTER*(MAX_LEN_MBUF) msg

      igate = -1
      IF ( gate.EQ.zeroRS ) igate = 0
      IF ( gate.EQ.oneRS ) igate = 1
      iactive = -1
      IF ( active.EQ.zeroRS ) iactive = 0
      IF ( active.EQ.oneRS ) iactive = 1
      WRITE(msg,'(A,11(1X,I0),1X,ES25.17E3)') '%WD_FACE_V1',
     & it,k,axis,i,j,side,tx,ty,myProcId,igate,iactive,volume*active
      CALL PRINT_MESSAGE( msg, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )
      RETURN
      END
