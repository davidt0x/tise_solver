        PROGRAM harmonic !  copied from  bf_sq_lan50.f  on Nov. 1, 2019
        implicit double precision (a-h,o-z)
	PARAMETER (nt=900)
        REAL*8 val(nt)
        real*8 vec(nt,nt),aham(nt,nt)
c23456789012345678901234567890123456789012345678901234567890123456789012
      pi = 4.0d0*datan(1.0d0)
      tpi = 2.0d0*pi
      pi2 = pi*pi
      sq2  = dsqrt(2.0d0)
      aa = 1.0d0   ! square well width is the unit of length
      omega = 50.0d0 ! in units of E_1 === (pi^2*hbar^2)/(2*m_0*a^2)
c some constants
      factanal = (pi*omega/2.0d0)**(0.25d0)
      expanal = 0.25d0*pi2*omega
      aa = 1.0d0   ! square well width
c construct the matrix
      nmax = nt
      do 1 n = 1,nmax
        aham(n,n)=n*n+pi2*omega*omega*(1.0d0 - 6.0d0/(pi*n)**2)/48.0d0
        do 2 m = 1,n-1
          nmm = n - m
          amm = ((-1)**(nmm) + 1.0d0)/dfloat(nmm)**2
          npm = n + m
          app = ((-1)**(npm) + 1.0d0)/dfloat(npm)**2
          aham(n,m) = 0.25d0*omega*omega*(amm - app)
          aham(m,n) = aham(n,m) ! symmetric
2       continue
1     continue
      na = nmax
      call eigen(na,aham,val,vec)
c write out eigenvalues
      write(16,*)'nmax,omega = ',nmax,omega
      write(17,*)'nmax,omega = ',nmax,omega
      write(18,*)'nmax,omega = ',nmax,omega
      do 3 n = 1,nmax
        ana = omega*(n-0.5d0) ! analytical answer
        write(16,96)n,n*n,val(n),ana
3     continue
96    format(5x,i5,1x,i10,2x,2(f16.6,1x))
c print out the ground state and 1st excited state wave function
c and probability density
      sum0 = 0.0d0
      sum1 = 0.0d0
      do 5 ix = 1,200
        xx = ix*0.005d0 ! in units of a
        psi0 = 0.0d0 ! ground state
        psi1 = 0.0d0 ! first excited state
        do 7 n = 1,nmax
          psi0 = psi0 + vec(n,1)*dsin(n*pi*xx)
          psi1 = psi1 + vec(n,2)*dsin(n*pi*xx)
7       continue
        psi0 = psi0*dsqrt(2.0d0)
        psi1 = psi1*dsqrt(2.0d0)
c
        dens0 = psi0*psi0
        dens1 = psi1*psi1
c check sum rule
        sum0 = sum0 + 0.005d0*dens0
        sum1 = sum1 + 0.005d0*dens1
        write(17,94)xx,psi0,dens0,sum0,psi1,dens1,sum1
c compare to analytical result
        anal0 = factanal*dexp(-expanal*(xx - 0.5d0)**2)
        anal1 = -anal0*2.0d0*dsqrt(expanal)*(xx - 0.5d0)
        write(18,95)xx,psi0,anal0,psi1,anal1
5     continue
94    format(5x,f9.4,1x,2(f14.8,1x,f14.8,2x,f14.8,3x))
95    format(5x,f9.4,1x,2(f14.8,1x,f14.8,5x))
      write(6,*)' sum0 = ',sum0
      write(6,*)' sum1 = ',sum1
      stop
      end
!--------------------------------------!
        SUBROUTINE eigen(n,a,val,vec)
!--------------------------------------!
c        implicit none
        implicit double precision (a-h,o-z)
        integer i,j,n,nt,n3,nmax,LDA,LWORK,LIWORK
c nov. 2, 2019
c        PARAMETER (nt=4,n3=4)                 
        PARAMETER (nt=900,n3=900)                 
        PARAMETER (NMAX=n3,LDA=NMAX)
        PARAMETER (LWORK=1+2*NMAX*NMAX+6*NMAX,LIWORK=3+5*NMAX)
        CHARACTER JOBZ,UPLO
        real*8 vec(nt,nt)
        real*8 val(nt)
        real*8 a(NMAX,NMAX),WORK(LWORK)
        INTEGER IWORK(LIWORK),INFO
        N=n3
        UPLO='U'
        JOBZ='V'
        IF (UPLO.EQ.'U') THEN
        END IF
c23456789012345678901234567890123456789012345678901234567890123456789012
        CALL dsyevd (JOBZ,UPLO,N,a,LDA,val,WORK,LWORK,IWORK,LIWORK,INFO)
        IF (INFO.GT.0)THEN
        WRITE (*,*) 'Failure to converge'                               
        ENDIF
        do 3 i=1,n
        do 3 j=1,n
    3    vec(i,j)=A(i,j)
        return
        end SUBROUTINE eigen
