C     Serial 8-by-7 Cartesian verification basin, one vertical level.
      INTEGER sNx,sNy,OLx,OLy,nSx,nSy,nPx,nPy,Nx,Ny,Nr
      PARAMETER ( sNx=8, sNy=7, OLx=2, OLy=2,
     &            nSx=1, nSy=1, nPx=1, nPy=1, Nr=1,
     &            Nx=sNx*nSx*nPx, Ny=sNy*nSy*nPy )
      INTEGER MAX_OLX,MAX_OLY
      PARAMETER ( MAX_OLX=OLx, MAX_OLY=OLy )
