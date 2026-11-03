#include "WETDRY_OPTIONS.h"

C     Test-only fault on the south copy of two closed Y faces.
C     Opposite signs cancel over rows of the ledger and over time.
      SUBROUTINE WD_AUDIT_TEST_FAULT( bi, bj, k, vTrans,
     I                               myIter )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GRID.h"
#include "WETDRY.h"
      INTEGER bi, bj, k, myIter, gx, gy, i, j, x
      _RL vTrans(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL bias

      IF ( myIter.LT.1 .OR. myIter.GT.2 .OR. k.NE.1 ) RETURN
      gx = myXGlobalLo - 1 + (bi-1)*sNx
      gy = myYGlobalLo - 1 + (bj-1)*sNy
      IF ( gy+sNy+1.NE.5 ) RETURN
      j = sNy+1
      DO i=1,sNx
       x = gx+i
       IF ( x.EQ.6 .OR. x.EQ.7 ) THEN
        IF ( wdMaskS(i,j,bi,bj).NE.zeroRS
     &       .OR. maskC(i,j-1,k,bi,bj).NE.oneRS
     &       .OR. maskC(i,j,k,bi,bj).NE.oneRS ) THEN
         STOP 'WD_AUDIT_TEST_FAULT: expected closed face with wet cells'
        ENDIF
        bias = 0.03125 _d 0
        IF ( x.EQ.7 ) bias = -bias
        IF ( myIter.EQ.2 ) bias = -bias
        vTrans(i,j) = vTrans(i,j) + bias
       ENDIF
      ENDDO
      RETURN
      END
