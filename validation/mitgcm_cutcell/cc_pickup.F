#include "CPP_OPTIONS.h"

      SUBROUTINE CC_INI_FIELDS(myThid)
C     Mode 2 carries its own prognostic state, so initialize the unused
C     stock fields from the original inputs even on a mode-2 restart.
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "CUT_CELL.h"
      INTEGER myThid,firstIter
      _RL firstTime
      IF (ccMode.EQ.2 .AND. nIter0.GT.0) THEN
       firstIter=nIter0
       firstTime=startTime
       nIter0=0
       startTime=baseTime
       CALL INI_FIELDS(myThid)
       nIter0=firstIter
       startTime=firstTime
      ELSE
       CALL INI_FIELDS(myThid)
      ENDIF
      END

      SUBROUTINE CC_PICKUP(reading,myIter,myTime,myThid)
C     Single portable global stream. Publish only after every record
C     has been gathered, written and closed. A different process grid
C     reads the same global records and extracts its own tile.
      USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT64
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "CUT_CELL.h"
      LOGICAL reading,root
      INTEGER myIter,myThid,unit,ios,k,i0,j0,ix,iy,iz,mode,iter
      INTEGER(INT64) nbytes
      _RL myTime,dt,g,time,local(sNx,sNy),global(Nx,Ny)
      CHARACTER*8 magic
      CHARACTER*40 filename,temporary
      CHARACTER*8 signature
      PARAMETER (signature='CCPICK01')
      WRITE(filename,'(A,I10.10,A)') 'cut_pickup.',myIter,'.bin'
      temporary=TRIM(filename)//'.tmp'
      root=myProcId.EQ.0
      i0=myXGlobalLo-1
      j0=myYGlobalLo-1
      IF (reading) THEN
       OPEN(NEWUNIT=unit,FILE=TRIM(filename),STATUS='OLD',
     &      ACCESS='STREAM',FORM='UNFORMATTED',ACTION='READ',IOSTAT=ios)
       IF (ios.NE.0) CALL CC_FAIL('missing '//TRIM(filename),myThid)
       INQUIRE(UNIT=unit,SIZE=nbytes)
       IF (nbytes.NE.52_INT64+80_INT64*Nx*Ny)
     &  CALL CC_FAIL('wrong cut-cell pickup size',myThid)
       READ(unit,IOSTAT=ios) magic,ix,iy,iz,mode,iter,dt,g,time
       IF (ios.NE.0) CALL CC_FAIL('invalid pickup header',myThid)
       IF (magic.NE.signature.OR.ix.NE.Nx.OR.iy.NE.Ny.OR.iz.NE.Nr
     &     .OR.mode.NE.ccMode.OR.iter.NE.myIter.OR.dt.NE.deltaTClock
     &     .OR.g.NE.gravity.OR..NOT.IEEE_IS_FINITE(time))
     &  CALL CC_FAIL('pickup does not match grid or clock',myThid)
       IF (ABS(time-myTime).GT.16. _d 0*EPSILON(time)
     &     *MAX(1. _d 0,ABS(time),ABS(myTime)))
     &  CALL CC_FAIL('pickup time does not match model clock',myThid)
      ELSEIF (root) THEN
       OPEN(NEWUNIT=unit,FILE=TRIM(temporary),STATUS='REPLACE',
     &      ACCESS='STREAM',FORM='UNFORMATTED',ACTION='WRITE',
     &      IOSTAT=ios)
       IF (ios.NE.0) CALL CC_FAIL('cannot create cut pickup',myThid)
       WRITE(unit,IOSTAT=ios) signature,Nx,Ny,Nr,ccMode,myIter,
     &                       deltaTClock,gravity,myTime
       IF (ios.NE.0) CALL CC_FAIL('cannot write pickup header',myThid)
      ENDIF

      DO k=1,10
       IF (reading) THEN
        READ(unit,IOSTAT=ios) global
        IF (ios.NE.0) CALL CC_FAIL('cannot read pickup field',myThid)
        IF (ANY(.NOT.IEEE_IS_FINITE(global)))
     &   CALL CC_FAIL('nonfinite cut-cell pickup field',myThid)
        local=global(i0+1:i0+sNx,j0+1:j0+sNy)
        IF (k.LE.4) THEN
         ccState(k,:,:)=local
        ELSEIF (k.EQ.5) THEN
         ccBed=local
        ELSEIF (ANY(local.NE.geometry(k))) THEN
         CALL CC_FAIL('cut-cell pickup geometry changed',myThid)
        ENDIF
       ELSE
        IF (k.LE.4) THEN
         local=ccState(k,:,:)
        ELSEIF (k.EQ.5) THEN
         local=ccBed
        ELSE
         local=geometry(k)
        ENDIF
        CALL GATHER_2D_R8(global,local,Nx,Ny,.FALSE.,.FALSE.,myThid)
        IF (root) THEN
         WRITE(unit,IOSTAT=ios) global
         IF (ios.NE.0) CALL CC_FAIL('cannot write pickup field',myThid)
        ENDIF
       ENDIF
      ENDDO
      IF (reading) THEN
       CLOSE(unit)
       IF (ccMode.EQ.1) THEN
        IF (ANY(ccArea.GT.0. _d 0 .AND.
     &      ccBed.NE.-drF(1)*hFacC(1:sNx,1:sNy,1,1,1)))
     &   CALL CC_FAIL('native pickup bathymetry changed',myThid)
       ENDIF
       IF (ANY(ccState(1,:,:).LT.0. _d 0) .OR.
     &     ANY(ccState(1,:,:).EQ.0. _d 0 .AND.
     &         ccState(4,:,:).NE.0. _d 0) .OR.
     &     ANY(SPREAD(ccArea.EQ.0. _d 0,1,4) .AND.
     &         ccState.NE.0. _d 0))
     &  CALL CC_FAIL('invalid conserved pickup state',myThid)
      ELSEIF (root) THEN
       CLOSE(unit,IOSTAT=ios)
       IF (ios.NE.0) CALL CC_FAIL('cannot close cut-cell pickup',myThid)
       CALL RENAME(TRIM(temporary),TRIM(filename),ios)
       IF (ios.NE.0) CALL CC_FAIL('cannot publish cut pickup',myThid)
      ENDIF
      CONTAINS

      FUNCTION geometry(record) RESULT(metric)
      INTEGER record
      _RL metric(sNx,sNy)
      SELECT CASE(record)
       CASE(6)
        metric=ccArea
       CASE(7)
        metric=ccWest(1:sNx,:)
       CASE(8)
        metric=ccWest(2:sNx+1,:)
       CASE(9)
        metric=ccSouth(:,1:sNy)
       CASE(10)
        metric=ccSouth(:,2:sNy+1)
      END SELECT
      END FUNCTION geometry
      END
