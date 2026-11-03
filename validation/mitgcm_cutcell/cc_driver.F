#include "CPP_OPTIONS.h"

      SUBROUTINE CC_SETUP( myThid )
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "EOS.h"
#include "CUT_CELL.h"
      INTEGER myThid,unit,ios,nbytes,i,j,edge
      LOGICAL enabled
      _RL fractions(sNx,sNy)
      NAMELIST /CUTCELL_PARM/ ccMode

      INQUIRE(FILE='data.cutcell',EXIST=enabled)
      IF (.NOT.enabled) THEN
       IF (myThid.EQ.1) ccMode=0
       _BARRIER
       RETURN
      ENDIF
      IF ( nThreads.NE.1 .OR. nSx*nSy*nPx*nPy.NE.1 .OR. Nr.NE.1 )
     & STOP 'CUTCELL requires a serial, one-level, one-tile grid'
      OPEN(NEWUNIT=unit,FILE='data.cutcell',STATUS='OLD',
     &     ACTION='READ',IOSTAT=ios)
      IF (ios.NE.0) STOP 'CUTCELL cannot open data.cutcell'
      ccMode=0
      READ(unit,NML=CUTCELL_PARM,IOSTAT=ios)
      CLOSE(unit)
      IF (ios.NE.0 .OR. ccMode.LT.1 .OR. ccMode.GT.2)
     & STOP 'CUTCELL requires ccMode=1 (linear) or 2 (shallow water)'
      IF ( .NOT.usingCartesianGrid .OR. nonHydrostatic .OR.
     &     .NOT.fluidIsWater .OR. nIter0.NE.0 .OR. nonlinFreeSurf.NE.0
     &     .OR. useOBCS .OR. useRealFreshWaterFlux .OR.
     &     momAdvection .OR. momViscosity .OR. momForcing .OR.
     &     tempStepping .OR. saltStepping .OR. f0.NE.0. _d 0 .OR.
     &     beta.NE.0. _d 0 ) STOP 'CUTCELL unsupported configuration'
      IF ( .NOT.IEEE_IS_FINITE(gravity) .OR. gravity.LE.0. _d 0 .OR.
     &     .NOT.IEEE_IS_FINITE(deltaTClock) .OR.
     &     deltaTClock.LE.0. _d 0 .OR. nTimeSteps.LE.0 )
     & STOP 'CUTCELL invalid gravity or model clock'
      IF (ccMode.EQ.1) THEN
       IF ( .NOT.exactConserv .OR. .NOT.implicitFreeSurface .OR.
     &      implicDiv2DFlow.NE.1. _d 0 .OR. implicSurfPress.NE.1. _d 0
     &      .OR. deltaTMom.NE.deltaTClock .OR.
     &      deltaTFreeSurf.NE.deltaTClock .OR. tAlpha.NE.0. _d 0
     &      .OR. sBeta.NE.0. _d 0 .OR. eosType.NE.'LINEAR'
     &      .OR. deepFacC(1).NE.1. _d 0 .OR. rhoFacC(1).NE.1. _d 0 )
     &   STOP 'CUTCELL requires Boussinesq implicit pressure'
      ENDIF

C     Raw IEEE binary64, in Fortran order. Lengths use unmodified MITgcm
C     grid metrics. The first/exterior face is geometrically present in
C     mode 2 but carries zero transport. Mode 1 requires a dry perimeter.
      OPEN(NEWUNIT=unit,FILE='cut_geometry.bin',STATUS='OLD',
     &     ACTION='READ',FORM='UNFORMATTED',ACCESS='STREAM',IOSTAT=ios)
      IF (ios.NE.0) STOP 'CUTCELL missing cut_geometry.bin'
      INQUIRE(UNIT=unit,SIZE=nbytes)
      IF (nbytes.NE.8*(3*sNx*sNy+sNx+sNy))
     & STOP 'CUTCELL wrong cut_geometry.bin size'
      READ(unit,IOSTAT=ios) fractions,ccWest,ccSouth
      CLOSE(unit)
      IF (ios.NE.0) STOP 'CUTCELL could not read geometry'
      IF ( ANY(.NOT.IEEE_IS_FINITE(fractions)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(ccWest)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(ccSouth)) .OR.
     &     ANY(fractions.LT.0. _d 0) .OR. ANY(fractions.GT.1. _d 0)
     &     .OR. ANY(ccWest.LT.0. _d 0) .OR. ANY(ccWest.GT.1. _d 0)
     &     .OR. ANY(ccSouth.LT.0. _d 0) .OR. ANY(ccSouth.GT.1. _d 0)
     &     .OR. .NOT.ANY(fractions.GT.0. _d 0) )
     & STOP 'CUTCELL invalid geometry fractions'
      DO j=1,sNy
       DO i=2,sNx
        IF (ccWest(i,j).GT.0. _d 0 .AND.
     &      MIN(fractions(i-1,j),fractions(i,j)).EQ.0. _d 0)
     &   STOP 'CUTCELL west aperture meets a solid cell'
       ENDDO
      ENDDO
      DO j=2,sNy
       DO i=1,sNx
        IF (ccSouth(i,j).GT.0. _d 0 .AND.
     &      MIN(fractions(i,j-1),fractions(i,j)).EQ.0. _d 0)
     &   STOP 'CUTCELL south aperture meets a solid cell'
       ENDDO
      ENDDO
      IF ( ANY(ccWest(1,:).GT.0. _d 0 .AND.
     &         fractions(1,:).EQ.0. _d 0) .OR.
     &     ANY(ccWest(sNx+1,:).GT.0. _d 0 .AND.
     &         fractions(sNx,:).EQ.0. _d 0) .OR.
     &     ANY(ccSouth(:,1).GT.0. _d 0 .AND.
     &         fractions(:,1).EQ.0. _d 0) .OR.
     &     ANY(ccSouth(:,sNy+1).GT.0. _d 0 .AND.
     &         fractions(:,sNy).EQ.0. _d 0) )
     & STOP 'CUTCELL exterior aperture meets a solid cell'

      DO j=1,sNy
       DO i=1,sNx
        ccArea(i,j)=rA(i,j,1,1)*fractions(i,j)
       ENDDO
       DO i=1,sNx+1
        edge=MIN(i,sNx)
        ccWest(i,j)=dyG(edge,j,1,1)*ccWest(i,j)
       ENDDO
      ENDDO
      DO j=1,sNy+1
       edge=MIN(j,sNy)
       DO i=1,sNx
        ccSouth(i,j)=dxG(i,edge,1,1)*ccSouth(i,j)
       ENDDO
      ENDDO
      IF (ccMode.EQ.2) RETURN

C     The stock implicit solver sees effective wet-cell area and open
C     face width. DXC/DYC remain the pressure-gradient distances. A
C     dry border suppresses the model's default periodic seam.
      IF ( ANY(ccArea(1,:).GT.0. _d 0) .OR.
     &     ANY(ccArea(sNx,:).GT.0. _d 0) .OR.
     &     ANY(ccArea(:,1).GT.0. _d 0) .OR.
     &     ANY(ccArea(:,sNy).GT.0. _d 0) )
     & STOP 'CUTCELL linear mode requires a dry border'
      DO j=1,sNy
       DO i=1,sNx
        IF ((ccArea(i,j).GT.0. _d 0) .NEQV.
     &      (hFacC(i,j,1,1,1).GT.0. _d 0))
     &   STOP 'CUTCELL cell fractions and native bathymetry disagree'
        IF ((ccWest(i,j).GT.0. _d 0) .NEQV.
     &      (hFacW(i,j,1,1,1).GT.0. _d 0))
     &   STOP 'CUTCELL west apertures and native masks disagree'
        IF ((ccSouth(i,j).GT.0. _d 0) .NEQV.
     &      (hFacS(i,j,1,1,1).GT.0. _d 0))
     &   STOP 'CUTCELL south apertures and native masks disagree'
        IF (ccArea(i,j).GT.0. _d 0) THEN
         rA(i,j,1,1)=ccArea(i,j)
         recip_rA(i,j,1,1)=1. _d 0/ccArea(i,j)
        ENDIF
        IF (ccWest(i,j).GT.0. _d 0) THEN
         dyG(i,j,1,1)=ccWest(i,j)
         recip_dyG(i,j,1,1)=1. _d 0/ccWest(i,j)
        ENDIF
        IF (ccSouth(i,j).GT.0. _d 0) THEN
         dxG(i,j,1,1)=ccSouth(i,j)
         recip_dxG(i,j,1,1)=1. _d 0/ccSouth(i,j)
        ENDIF
       ENDDO
      ENDDO
      _EXCH_XY_RS(rA,myThid)
      _EXCH_XY_RS(recip_rA,myThid)
      CALL EXCH_UV_XY_RS(dyG,dxG,.FALSE.,myThid)
      CALL EXCH_UV_XY_RS(recip_dyG,recip_dxG,.FALSE.,myThid)
      END

      SUBROUTINE CC_FORWARD_STEP( iloop,myTime,myIter,myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "CUT_CELL.h"
      INTEGER iloop,myIter,myThid,i,j
      _RL myTime,west(sNx+1,sNy),south(sNx,sNy+1)

      IF (ccMode.EQ.0) THEN
       CALL FORWARD_STEP(iloop,myTime,myIter,myThid)
       RETURN
      ENDIF
      IF (iloop.EQ.1) THEN
       CALL CC_INITIAL_STATE
       CALL CC_WRITE_STATE(nIter0,myThid)
      ENDIF
      IF (ccMode.EQ.1) THEN
       CALL FORWARD_STEP(iloop,myTime,myIter,myThid)
       west=0. _d 0
       south=0. _d 0
       DO j=1,sNy
        DO i=1,sNx
         west(i,j)=ccWest(i,j)*drF(1)*hFacW(i,j,1,1,1)
     &      *uVel(i,j,1,1,1)*deltaTClock
         south(i,j)=ccSouth(i,j)*drF(1)*hFacS(i,j,1,1,1)
     &      *vVel(i,j,1,1,1)*deltaTClock
        ENDDO
       ENDDO
       CALL CC_TRANSFER(sNx,sNy,ccState,west,south)
      ELSE
       CALL CC_SWE_ADVANCE(sNx,sNy,ccArea,ccWest,ccSouth,ccBed,
     &                     gravity,deltaTClock,ccState)
       myIter=nIter0+iloop
       myTime=startTime+deltaTClock*iloop
      ENDIF
      CALL CC_WRITE_STATE(myIter,myThid)
      WRITE(standardMessageUnit,'(A,I8,1X,ES24.16)')
     & 'CUTCELL clock ',myIter,myTime
      END

      SUBROUTINE CC_INITIAL_STATE
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "CUT_CELL.h"
      INTEGER unit,ios,nbytes,i,j
      _RL depth(sNx,sNy),ux(sNx,sNy),uy(sNx,sNy),tracer(sNx,sNy)
      OPEN(NEWUNIT=unit,FILE='cut_initial.bin',STATUS='OLD',
     &     ACTION='READ',FORM='UNFORMATTED',ACCESS='STREAM',IOSTAT=ios)
      IF (ios.NE.0) STOP 'CUTCELL missing cut_initial.bin'
      INQUIRE(UNIT=unit,SIZE=nbytes)
      IF (nbytes.NE.8*5*sNx*sNy) STOP 'CUTCELL wrong initial file size'
      READ(unit,IOSTAT=ios) ccBed,depth,ux,uy,tracer
      CLOSE(unit)
      IF (ios.NE.0) STOP 'CUTCELL could not read initial state'
      IF ( ANY(.NOT.IEEE_IS_FINITE(ccBed)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(depth)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(ux)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(uy)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(tracer)) .OR.
     &     ANY(depth.LT.0. _d 0) ) STOP 'CUTCELL invalid initial state'
      ccState=0. _d 0
      DO j=1,sNy
       DO i=1,sNx
        IF (ccArea(i,j).GT.0. _d 0) THEN
         IF (ccMode.EQ.1) THEN
          ccBed(i,j)=-drF(1)*hFacC(i,j,1,1,1)
          depth(i,j)=etaN(i,j,1,1)-ccBed(i,j)
          ux(i,j)=0. _d 0
          uy(i,j)=0. _d 0
         ENDIF
         ccState(1,i,j)=ccArea(i,j)*depth(i,j)
         ccState(2,i,j)=ccState(1,i,j)*ux(i,j)
         ccState(3,i,j)=ccState(1,i,j)*uy(i,j)
         ccState(4,i,j)=ccState(1,i,j)*tracer(i,j)
        ENDIF
       ENDDO
      ENDDO
      IF ( ANY(ccState(1,:,:).LT.0. _d 0) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(ccState)) )
     & STOP 'CUTCELL nonrepresentable conserved state'
      END

      SUBROUTINE CC_WRITE_STATE(myIter,myThid)
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "CUT_CELL.h"
      INTEGER myIter,myThid,i,j,k
      _RL field(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      CHARACTER*8 names(4)
      DATA names /'CCDepth ','CCQx    ','CCQy    ','CCTracer'/
      DO k=1,4
       IF (ccMode.EQ.2 .OR. k.EQ.1 .OR. k.EQ.4) THEN
        field=0. _d 0
        DO j=1,sNy
         DO i=1,sNx
          IF (ccArea(i,j).GT.0. _d 0) THEN
           IF (k.LE.3) THEN
            field(i,j,1,1)=ccState(k,i,j)/ccArea(i,j)
           ELSEIF (ccState(1,i,j).GT.0. _d 0) THEN
            field(i,j,1,1)=ccState(4,i,j)/ccState(1,i,j)
           ENDIF
          ENDIF
         ENDDO
        ENDDO
        CALL WRITE_FLD_XY_RL(names(k),'I10',field,myIter,myThid)
       ENDIF
      ENDDO
      END
