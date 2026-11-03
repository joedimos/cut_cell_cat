C     Experiment-local geometry and conserved state on the single tile.
C     State entries are water volume, x/y discharge times area, and
C     tracer content. ccWest and ccSouth contain geometric lengths.
      INTEGER ccMode,ccPickupEvery
      LOGICAL ccWriteFluxes
      _RL ccArea(sNx,sNy),ccWest(sNx+1,sNy),ccSouth(sNx,sNy+1)
      _RL ccBed(sNx,sNy),ccState(4,sNx,sNy)
      COMMON /CC_MODE/ ccMode,ccPickupEvery,ccWriteFluxes
      COMMON /CC_GEOMETRY/ ccArea,ccWest,ccSouth,ccBed
      COMMON /CC_STATE/ ccState
