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
      INTEGER myThid,unit,ios,nbytes,i,j,edge,i0,j0
      LOGICAL enabled
      _RL fractions(Nx,Ny),globalW(Nx+1,Ny),globalS(Nx,Ny+1)
      NAMELIST /CUTCELL_PARM/ ccMode,ccWriteFluxes,ccPickupEvery

      INQUIRE(FILE='data.cutcell',EXIST=enabled)
      IF (.NOT.enabled) THEN
       IF (myThid.EQ.1) ccMode=0
       _BARRIER
       RETURN
      ENDIF
      IF (nThreads.NE.1 .OR. nSx*nSy.NE.1)
     & CALL CC_FAIL('one thread and tile per rank required',myThid)
      OPEN(NEWUNIT=unit,FILE='data.cutcell',STATUS='OLD',
     &     ACTION='READ',IOSTAT=ios)
      IF (ios.NE.0) CALL CC_FAIL('cannot open data.cutcell',myThid)
      ccMode=0
      ccWriteFluxes=.FALSE.
      ccPickupEvery=0
      READ(unit,NML=CUTCELL_PARM,IOSTAT=ios)
      CLOSE(unit)
      IF (ios.NE.0 .OR. ccMode.LT.1 .OR. ccMode.GT.3
     &    .OR. ccPickupEvery.LT.0)
     & CALL CC_FAIL('invalid cut-cell mode or pickup interval',myThid)
      IF ( .NOT.usingCartesianGrid .OR. useCubedSphereExchange .OR.
     &     nonHydrostatic .OR.
     &     .NOT.fluidIsWater .OR. useOBCS .OR. useRealFreshWaterFlux
     &     .OR. momAdvection .OR. momViscosity .OR. momForcing .OR.
     &     f0.NE.0. _d 0 .OR. beta.NE.0. _d 0 )
     & CALL CC_FAIL('unsupported native configuration',myThid)
      IF ( .NOT.IEEE_IS_FINITE(gravity) .OR. gravity.LE.0. _d 0 .OR.
     &     .NOT.IEEE_IS_FINITE(deltaTClock) .OR.
     &     deltaTClock.LE.0. _d 0 .OR. nTimeSteps.LE.0 )
     & CALL CC_FAIL('invalid gravity or model clock',myThid)
      IF (ccMode.LE.2) THEN
       IF (Nr.NE.1 .OR. nonlinFreeSurf.NE.0 .OR.
     &     tempStepping .OR. saltStepping)
     &  CALL CC_FAIL('overlay modes require one level',myThid)
       IF (ccMode.EQ.2 .AND. pickupSuff.NE.' ')
     &  CALL CC_FAIL('mode 2 uses nIter0 to select pickups',myThid)
      ELSE
       IF (Nr.LT.2 .OR. nonlinFreeSurf.NE.4 .OR. select_rStar.NE.2 .OR.
     &     .NOT.staggerTimeStep .OR. ccWriteFluxes .OR.
     &     ccPickupEvery.NE.0)
     &  CALL CC_FAIL('mode 3 requires native rStar and pickups',myThid)
      ENDIF
      IF (ccMode.NE.2) THEN
       IF ( .NOT.exactConserv .OR. .NOT.implicitFreeSurface .OR.
     &      implicDiv2DFlow.NE.1. _d 0 .OR. implicSurfPress.NE.1. _d 0
     &      .OR. deltaTMom.NE.deltaTClock .OR.
     &      deltaTFreeSurf.NE.deltaTClock .OR.
     &      ANY(deepFacC.NE.1. _d 0) .OR. ANY(rhoFacC.NE.1. _d 0) )
     &  CALL CC_FAIL('requires Boussinesq implicit pressure',myThid)
       IF (ccMode.EQ.1 .AND. (tAlpha.NE.0. _d 0 .OR.
     &      sBeta.NE.0. _d 0 .OR. eosType.NE.'LINEAR'))
     &  CALL CC_FAIL('legacy mode requires a neutral EOS',myThid)
      ENDIF

C     Raw IEEE binary64, in Fortran order. Lengths use unmodified MITgcm
C     grid metrics. The first/exterior face is geometrically present in
C     mode 2 but carries zero transport. Mode 1 requires a dry perimeter.
      OPEN(NEWUNIT=unit,FILE='cut_geometry.bin',STATUS='OLD',
     &     ACTION='READ',FORM='UNFORMATTED',ACCESS='STREAM',IOSTAT=ios)
      IF (ios.NE.0)
     & CALL CC_FAIL('missing cut_geometry.bin',myThid)
      INQUIRE(UNIT=unit,SIZE=nbytes)
      IF (nbytes.NE.8*(3*Nx*Ny+Nx+Ny))
     & CALL CC_FAIL('wrong cut_geometry.bin size',myThid)
      READ(unit,IOSTAT=ios) fractions,globalW,globalS
      CLOSE(unit)
      IF (ios.NE.0)
     & CALL CC_FAIL('could not read geometry',myThid)
      IF ( ANY(.NOT.IEEE_IS_FINITE(fractions)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(globalW)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(globalS)) .OR.
     &     ANY(fractions.LT.0. _d 0) .OR. ANY(fractions.GT.1. _d 0)
     &     .OR. ANY(globalW.LT.0. _d 0) .OR. ANY(globalW.GT.1. _d 0)
     &     .OR. ANY(globalS.LT.0. _d 0) .OR. ANY(globalS.GT.1. _d 0)
     &     .OR. .NOT.ANY(fractions.GT.0. _d 0) )
     & CALL CC_FAIL('invalid geometry fractions',myThid)
      DO j=1,Ny
       DO i=2,Nx
        IF (globalW(i,j).GT.0. _d 0 .AND.
     &      MIN(fractions(i-1,j),fractions(i,j)).EQ.0. _d 0)
     &   CALL CC_FAIL('west aperture meets a solid cell',myThid)
       ENDDO
      ENDDO
      DO j=2,Ny
       DO i=1,Nx
        IF (globalS(i,j).GT.0. _d 0 .AND.
     &      MIN(fractions(i,j-1),fractions(i,j)).EQ.0. _d 0)
     & CALL CC_FAIL('south aperture meets a solid cell',myThid)
       ENDDO
      ENDDO
      IF ( ANY(globalW(1,:).GT.0. _d 0 .AND.
     &         fractions(1,:).EQ.0. _d 0) .OR.
     &     ANY(globalW(Nx+1,:).GT.0. _d 0 .AND.
     &         fractions(Nx,:).EQ.0. _d 0) .OR.
     &     ANY(globalS(:,1).GT.0. _d 0 .AND.
     &         fractions(:,1).EQ.0. _d 0) .OR.
     &     ANY(globalS(:,Ny+1).GT.0. _d 0 .AND.
     &         fractions(:,Ny).EQ.0. _d 0) )
     & CALL CC_FAIL('exterior aperture meets a solid cell',myThid)

      i0=myXGlobalLo-1
      j0=myYGlobalLo-1
      ccWest=globalW(i0+1:i0+sNx+1,j0+1:j0+sNy)
      ccSouth=globalS(i0+1:i0+sNx,j0+1:j0+sNy+1)
      DO j=1,sNy
       DO i=1,sNx
        ccArea(i,j)=rA(i,j,1,1)*fractions(i0+i,j0+j)
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
      IF ( ANY(fractions(1,:).GT.0. _d 0) .OR.
     &     ANY(fractions(Nx,:).GT.0. _d 0) .OR.
     &     ANY(fractions(:,1).GT.0. _d 0) .OR.
     &     ANY(fractions(:,Ny).GT.0. _d 0) )
     & CALL CC_FAIL('stock modes require a dry border',myThid)
      DO j=1,sNy
       DO i=1,sNx
        IF ((ccArea(i,j).GT.0. _d 0) .NEQV.
     &      ANY(hFacC(i,j,:,1,1).GT.0. _d 0))
     & CALL CC_FAIL('cell fractions and bathymetry disagree',myThid)
        IF ((ccWest(i,j).GT.0. _d 0) .NEQV.
     &      ANY(hFacW(i,j,:,1,1).GT.0. _d 0))
     & CALL CC_FAIL('west apertures and native masks disagree',myThid)
        IF ((ccSouth(i,j).GT.0. _d 0) .NEQV.
     &      ANY(hFacS(i,j,:,1,1).GT.0. _d 0))
     & CALL CC_FAIL('south apertures and native masks disagree',myThid)
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

      SUBROUTINE CC_FAIL(message,myThid)
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "EESUPPORT.h"
      CHARACTER*(*) message
      INTEGER myThid
      CHARACTER*(MAX_LEN_MBUF) msgBuf
#ifdef ALLOW_USE_MPI
      INTEGER ierr
#endif
      msgBuf='CUTCELL: '//message
      CALL PRINT_ERROR(msgBuf,myThid)
      FLUSH(errorMessageUnit)
#ifdef ALLOW_USE_MPI
      IF (usingMPI) CALL MPI_ABORT(MPI_COMM_MODEL,1,ierr)
#endif
      STOP 1
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
      _RL ledger(4,sNx,sNy)

      IF (ccMode.EQ.0 .OR. ccMode.EQ.3) THEN
       CALL FORWARD_STEP(iloop,myTime,myIter,myThid)
       RETURN
      ENDIF
      IF (iloop.EQ.1) THEN
       IF (nIter0.GT.0) THEN
        CALL CC_PICKUP(.TRUE.,nIter0,startTime,myThid)
       ELSE
        CALL CC_INITIAL_STATE(myThid)
       ENDIF
       CALL CC_WRITE_STATE(nIter0,myThid)
      ENDIF
      IF (ccWriteFluxes) ledger=0. _d 0
      IF (ccMode.EQ.1) THEN
       CALL FORWARD_STEP(iloop,myTime,myIter,myThid)
C      Bring the corrected velocity on the east/north seam into halo.
       CALL EXCH_UV_3D_RL(uVel,vVel,.FALSE.,Nr,myThid)
       DO j=1,sNy
        DO i=1,sNx+1
         west(i,j)=ccWest(i,j)*drF(1)*hFacW(i,j,1,1,1)
     &      *uVel(i,j,1,1,1)*deltaTClock
        ENDDO
       ENDDO
       DO j=1,sNy+1
        DO i=1,sNx
         south(i,j)=ccSouth(i,j)*drF(1)*hFacS(i,j,1,1,1)
     &      *vVel(i,j,1,1,1)*deltaTClock
        ENDDO
       ENDDO
       CALL CC_TRANSFER(ccState,west,south,
     &                  ccWriteFluxes,ledger,myThid)
      ELSE
       CALL CC_SWE_ADVANCE(ccArea,ccWest,ccSouth,ccBed,
     &                     gravity,deltaTClock,ccState,
     &                     ccWriteFluxes,ledger,myThid)
       myIter=nIter0+iloop
       myTime=startTime+deltaTClock*iloop
      ENDIF
      CALL CC_WRITE_STATE(myIter,myThid)
      IF (ccWriteFluxes) CALL CC_WRITE_FLUXES(ledger,myIter,myThid)
      IF (ccPickupEvery.GT.0) THEN
       IF (MOD(myIter,ccPickupEvery).EQ.0 .OR. iloop.EQ.nTimeSteps)
     &  CALL CC_PICKUP(.FALSE.,myIter,myTime,myThid)
      ENDIF
      WRITE(standardMessageUnit,'(A,I8,1X,ES24.16)')
     & 'CUTCELL clock ',myIter,myTime
      END

      SUBROUTINE CC_WRITE_FLUXES(ledger,myIter,myThid)
C     Signed transfers over one MITgcm clock interval. West/eastward
C     and south/northward follow native MDS indexing. Exterior faces
C     are closed, including the unrecorded terminal east/north faces.
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      INTEGER myIter,myThid,k
      _RL ledger(4,sNx,sNy)
      _RL field(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      CHARACTER*6 names(4)
      DATA names /'CCVolW','CCVolS','CCTrW ','CCTrS '/
      field=0. _d 0
      DO k=1,4
       field(1:sNx,1:sNy,1,1)=ledger(k,:,:)
       CALL WRITE_FLD_XY_RL(names(k),'I10',field,myIter,myThid)
      ENDDO
      END

      SUBROUTINE CC_INITIAL_STATE(myThid)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "CUT_CELL.h"
      INTEGER myThid,unit,ios,nbytes,i,j,i0,j0
      _RL depth(sNx,sNy),ux(sNx,sNy),uy(sNx,sNy),tracer(sNx,sNy)
      _RL initial(Nx,Ny,5)
      OPEN(NEWUNIT=unit,FILE='cut_initial.bin',STATUS='OLD',
     &     ACTION='READ',FORM='UNFORMATTED',ACCESS='STREAM',IOSTAT=ios)
      IF (ios.NE.0) CALL CC_FAIL('missing cut_initial.bin',myThid)
      INQUIRE(UNIT=unit,SIZE=nbytes)
      IF (nbytes.NE.8*5*Nx*Ny)
     & CALL CC_FAIL('wrong initial file size',myThid)
      READ(unit,IOSTAT=ios) initial
      CLOSE(unit)
      IF (ios.NE.0) CALL CC_FAIL('cannot read initial state',myThid)
      i0=myXGlobalLo-1
      j0=myYGlobalLo-1
      ccBed=initial(i0+1:i0+sNx,j0+1:j0+sNy,1)
      depth=initial(i0+1:i0+sNx,j0+1:j0+sNy,2)
      ux=initial(i0+1:i0+sNx,j0+1:j0+sNy,3)
      uy=initial(i0+1:i0+sNx,j0+1:j0+sNy,4)
      tracer=initial(i0+1:i0+sNx,j0+1:j0+sNy,5)
      IF ( ANY(.NOT.IEEE_IS_FINITE(ccBed)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(depth)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(ux)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(uy)) .OR.
     &     ANY(.NOT.IEEE_IS_FINITE(tracer)) .OR.
     &     ANY(depth.LT.0. _d 0) )
     & CALL CC_FAIL('invalid initial state',myThid)
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
     & CALL CC_FAIL('nonrepresentable conserved state',myThid)
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
