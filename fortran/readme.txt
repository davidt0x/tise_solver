To compile harmonic.f I used
$gfortran harmonic.f -llapack -lblas -o harm.exe

then I used
$time ./harm.exe

and I get:
  sum0 =    1.0000000000000020     
  sum1 =    1.0000000000000000     
Note: The following floating-point exceptions are signalling: IEEE_DENORMAL

real	0m0.128s
user	0m0.218s
sys	0m0.029s


I check the sum rule on the two first wave functions

fort.16 contains 4 columns of numbers
n (quantum #), n*n (expected result for empty infinite square well),  numerical eigenvalues,   analytical eigenvalues  

fort.17 contains the 1st 2 wave functions and their squares, as a function of x/a:

xx    psi_0     psi_0**2    sum_0      psi_1   psi_1**2    sum_1



fort.18 contains the 1st 2 wave functions and their analytical counterparts, as a function of x/a:

xx    psi_0   psi_0  psi_{0,analytical}      psi_1   psi_{1,analytical}


you can change omega and the matrix size (I don't know how to use the lapack routines so
I just change the array declarations in the main program and in the subroutine, so 900 ---> 2000, or
whatever, in 3 places) 
