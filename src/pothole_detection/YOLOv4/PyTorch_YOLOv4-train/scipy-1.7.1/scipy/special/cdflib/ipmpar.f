      INTEGER FUNCTION ipmpar(i)
C-----------------------------------------------------------------------
C
C     IPMPAR PROVIDES THE INTEGER MACHINE CONSTANTS FOR THE COMPUTER
C     THAT IS USED. IT IS ASSUMED THAT THE ARGUMENT I IS AN INTEGER
C     HAVING ONE OF THE VALUES 1-10. IPMPAR(I) HAS THE VALUE ...
C
C  INTEGERS.
C
C     ASSUME INTEGERS ARE REPRESENTED IN THE N-DIGIT, BASE-A FORM
C
C               SIGN ( X(N-1)*A**(N-1) + ... + X(1)*A + X(0) )
C
C               WHERE 0 .LE. X(I) .LT. A FOR I=0,...,N-1.
C
C     IPMPAR(1) = A, THE BASE.
C
C     IPMPAR(2) = N, THE NUMBER OF BASE-A DIGITS.
C
C     IPMPAR(3) = A**N - 1, THE LARGEST MAGNITUDE.
C
C  FLOATING-POINT NUMBERS.
C
C     IT IS ASSUMED THAT THE SINGLE AND DOUBLE PRECISION FLOATING
C     POINT ARITHMETICS HAVE THE SAME BASE, SAY B, AND THAT THE
C     NONZERO NUMBERS ARE REPRESENTED IN THE FORM
C
C               SIGN (B**E) * (X(1)/B + ... + X(M)/B**M)
C
C               WHERE X(I) = 0,1,...,B-1 FOR I=1,...,M,
C               X(1) .GE. 1, AND EMIN .LE. E .LE. EMAX.
C
C     IPMPAR(4) = B, THE BASE.
C
C  SINGLE-PRECISION
C
C     IPMPAR(5) = M, THE NUMBER OF BASE-B DIGITS.
C
C     IPMPAR(6) = EMIN, THE SMALLEST EXPONENT E.
C
C     IPMPAR(7) = EMAX, THE LARGEST EXPONENT E.
C
C  DOUBLE-PRECISION
C
C     IPMPAR(8) = M, THE NUMBER OF BASE-B DIGITS.
C
C     IPMPAR(9) = EMIN, THE SMALLEST EXPONENT E.
C
C     IPMPAR(10) = EMAX, THE LARGEST EXPONENT E.
C
C-----------------------------------------------------------------------
C
C     TO DEFINE THIS FUNCTION FOR THE COMPUTER BEING USED, ACTIVATE
C     THE DATA STATEMENTS FOR THE COMPUTER BY REMOVING THE C FROM
C     COLUMN 1. (ALL THE OTHER DATA STATEMENTS SHOULD HAVE C IN
C     COLUMN 1.)
C
C-----------------------------------------------------------------------
C
C     IPMPAR IS AN ADAPTATION OF THE FUNCTION I1MACH, WRITTEN BY
C     P.A. FOX, A.D. HALL, AND N.L. SCHRYER (BELL LABORATORIES).
C     IPMPAR WAS FORMED BY A.H. MORRIS (NSWC). THE CONSTANTS ARE
C     FROM BELL LABORATORIES, NSWC, AND OTHER SOURCES.
C
C-----------------------------------------------------------------------
C     .. Scalar Arguments ..
      INTEGER i
C     ..
C     .. Local Arrays ..
      INTEGER imach(10)
C     ..
C     .. Data statements ..
C
C     MACHINE CONSTANTS FOR AMDAHL MACHINES.
C
C     DATA IMACH( 1) /   2 /
C     DATA IMACH( 2) /  31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /  16 /
C     DATA IMACH( 5) /   6 /
C     DATA IMACH( 6) / -64 /
C     DATA IMACH( 7) /  63 /
C     DATA IMACH( 8) /  14 /
C     DATA IMACH( 9) / -64 /
C     DATA IMACH(10) /  63 /
C
C     MACHINE CONSTANTS FOR THE AT&T 3B SERIES, AT&T
C     PC 7300, AND AT&T 6300.
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    24 /
C     DATA IMACH( 6) /  -125 /
C     DATA IMACH( 7) /   128 /
C     DATA IMACH( 8) /    53 /
C     DATA IMACH( 9) / -1021 /
C     DATA IMACH(10) /  1024 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 1700 SYSTEM.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   33 /
C     DATA IMACH( 3) / 8589934591 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   24 /
C     DATA IMACH( 6) / -256 /
C     DATA IMACH( 7) /  255 /
C     DATA IMACH( 8) /   60 /
C     DATA IMACH( 9) / -256 /
C     DATA IMACH(10) /  255 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 5700 SYSTEM.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   39 /
C     DATA IMACH( 3) / 549755813887 /
C     DATA IMACH( 4) /    8 /
C     DATA IMACH( 5) /   13 /
C     DATA IMACH( 6) /  -50 /
C     DATA IMACH( 7) /   76 /
C     DATA IMACH( 8) /   26 /
C     DATA IMACH( 9) /  -50 /
C     DATA IMACH(10) /   76 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 6700/7700 SYSTEMS.
C
C     DATA IMACH( 1) /      2 /
C     DATA IMACH( 2) /     39 /
C     DATA IMACH( 3) / 549755813887 /
C     DATA IMACH( 4) /      8 /
C     DATA IMACH( 5) /     13 /
C     DATA IMACH( 6) /    -50 /
C     DATA IMACH( 7) /     76 /
C     DATA IMACH( 8) /     26 /
C     DATA IMACH( 9) / -32754 /
C     DATA IMACH(10) /  32780 /
C
C     MACHINE CONSTANTS FOR THE CDC 6000/7000 SERIES
C     60 BIT ARITHMETIC, AND THE CDC CYBER 995 64 BIT
C     ARITHMETIC (NOS OPERATING SYSTEM).
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   48 /
C     DATA IMACH( 3) / 281474976710655 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   48 /
C     DATA IMACH( 6) / -974 /
C     DATA IMACH( 7) / 1070 /
C     DATA IMACH( 8) /   95 /
C     DATA IMACH( 9) / -926 /
C     DATA IMACH(10) / 1070 /
C
C     MACHINE CONSTANTS FOR THE CDC CYBER 995 64 BIT
C     ARITHMETIC (NOS/VE OPERATING SYSTEM).
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    63 /
C     DATA IMACH( 3) / 9223372036854775807 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    48 /
C     DATA IMACH( 6) / -4096 /
C     DATA IMACH( 7) /  4095 /
C     DATA IMACH( 8) /    96 /
C     DATA IMACH( 9) / -4096 /
C     DATA IMACH(10) /  4095 /
C
C     MACHINE CONSTANTS FOR THE CRAY 1, XMP, 2, AND 3.
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    63 /
C     DATA IMACH( 3) / 9223372036854775807 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    47 /
C     DATA IMACH( 6) / -8189 /
C     DATA IMACH( 7) /  8190 /
C     DATA IMACH( 8) /    94 /
C     DATA IMACH( 9) / -8099 /
C     DATA IMACH(10) /  8190 /
C
C     MACHINE CONSTANTS FOR THE DATA GENERAL ECLIPSE S/200.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   15 /
C     DATA IMACH( 3) / 32767 /
C     DATA IMACH( 4) /   16 /
C     DATA IMACH( 5) /    6 /
C     DATA IMACH( 6) /  -64 /
C     DATA IMACH( 7) /   63 /
C     DATA IMACH( 8) /   14 /
C     DATA IMACH( 9) /  -64 /
C     DATA IMACH(10) /   63 /
C
C     MACHINE CONSTANTS FOR THE HARRIS 220.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   23 /
C     DATA IMACH( 3) / 8388607 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   23 /
C     DATA IMACH( 6) / -127 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   38 /
C     DATA IMACH( 9) / -127 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE HONEYWELL 600/6000
C     AND DPS 8/70 SERIES.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   35 /
C     DATA IMACH( 3) / 34359738367 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   27 /
C     DATA IMACH( 6) / -127 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   63 /
C     DATA IMACH( 9) / -127 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE HP 2100
C     3 WORD DOUBLE PRECISION OPTION WITH FTN4
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   15 /
C     DATA IMACH( 3) / 32767 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   23 /
C     DATA IMACH( 6) / -128 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   39 /
C     DATA IMACH( 9) / -128 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE HP 2100
C     4 WORD DOUBLE PRECISION OPTION WITH FTN4
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   15 /
C     DATA IMACH( 3) / 32767 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   23 /
C     DATA IMACH( 6) / -128 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   55 /
C     DATA IMACH( 9) / -128 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE HP 9000.
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    24 /
C     DATA IMACH( 6) /  -126 /
C     DATA IMACH( 7) /   128 /
C     DATA IMACH( 8) /    53 /
C     DATA IMACH( 9) / -1021 /
C     DATA IMACH(10) /  1024 /
C
C     MACHINE CONSTANTS FOR THE IBM 360/370 SERIES,
C     THE ICL 2900, THE ITEL AS/6, THE XEROX SIGMA
C     5/7/9 AND THE SEL SYSTEMS 85/86.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /   16 /
C     DATA IMACH( 5) /    6 /
C     DATA IMACH( 6) /  -64 /
C     DATA IMACH( 7) /   63 /
C     DATA IMACH( 8) /   14 /
C     DATA IMACH( 9) /  -64 /
C     DATA IMACH(10) /   63 /
C
C     MACHINE CONSTANTS FOR THE IBM PC.
C
C      DATA imach(1)/2/
C      DATA imach(2)/31/
C      DATA imach(3)/2147483647/
C      DATA imach(4)/2/
C      DATA imach(5)/24/
C      DATA imach(6)/-125/
C      DATA imach(7)/128/
C      DATA imach(8)/53/
C      DATA imach(9)/-1021/
C      DATA imach(10)/1024/
C
C     MACHINE CONSTANTS FOR THE MACINTOSH II - ABSOFT
C     MACFORTRAN II.
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    24 /
C     DATA IMACH( 6) /  -125 /
C     DATA IMACH( 7) /   128 /
C     DATA IMACH( 8) /    53 /
C     DATA IMACH( 9) / -1021 /
C     DATA IMACH(10) /  1024 /
C
C     MACHINE CONSTANTS FOR THE MICROVAX - VMS FORTRAN.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   24 /
C     DATA IMACH( 6) / -127 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   56 /
C     DATA IMACH( 9) / -127 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KA PROCESSOR).
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   35 /
C     DATA IMACH( 3) / 34359738367 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   27 /
C     DATA IMACH( 6) / -128 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   54 /
C     DATA IMACH( 9) / -101 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KI PROCESSOR).
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   35 /
C     DATA IMACH( 3) / 34359738367 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   27 /
C     DATA IMACH( 6) / -128 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   62 /
C     DATA IMACH( 9) / -128 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE PDP-11 FORTRAN SUPPORTING
C     32-BIT INTEGER ARITHMETIC.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   24 /
C     DATA IMACH( 6) / -127 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   56 /
C     DATA IMACH( 9) / -127 /
C     DATA IMACH(10) /  127 /
C
C     MACHINE CONSTANTS FOR THE SEQUENT BALANCE 8000.
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    24 /
C     DATA IMACH( 6) /  -125 /
C     DATA IMACH( 7) /   128 /
C     DATA IMACH( 8) /    53 /
C     DATA IMACH( 9) / -1021 /
C     DATA IMACH(10) /  1024 /
C
C     MACHINE CONSTANTS FOR THE SILICON GRAPHICS IRIS-4D
C     SERIES (MIPS R3000 PROCESSOR).
C
C     DATA IMACH( 1) /     2 /
C     DATA IMACH( 2) /    31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /     2 /
C     DATA IMACH( 5) /    24 /
C     DATA IMACH( 6) /  -125 /
C     DATA IMACH( 7) /   128 /
C     DATA IMACH( 8) /    53 /
C     DATA IMACH( 9) / -1021 /
C     DATA IMACH(10) /  1024 /
C
C     MACHINE CONSTANTS FOR IEEE ARITHMETIC MACHINES, SUCH AS THE AT&T
C     3B SERIES, MOTOROLA 68000 BASED MACHINES (E.G. SUN 3 AND AT&T
C     PC 7300), AND 8087 BASED MICROS (E.G. IBM PC AND AT&T 6300).
C
      DATA IMACH( 1) /     2 /
      DATA IMACH( 2) /    31 /
      DATA IMACH( 3) / 2147483647 /
      DATA IMACH( 4) /     2 /
      DATA IMACH( 5) /    24 /
      DATA IMACH( 6) /  -125 /
      DATA IMACH( 7) /   128 /
      DATA IMACH( 8) /    53 /
      DATA IMACH( 9) / -1021 /
      DATA IMACH(10) /  1024 /
C
C     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   35 /
C     DATA IMACH( 3) / 34359738367 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   27 /
C     DATA IMACH( 6) / -128 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   60 /
C     DATA IMACH( 9) /-1024 /
C     DATA IMACH(10) / 1023 /
C
C     MACHINE CONSTANTS FOR THE VAX 11/780.
C
C     DATA IMACH( 1) /    2 /
C     DATA IMACH( 2) /   31 /
C     DATA IMACH( 3) / 2147483647 /
C     DATA IMACH( 4) /    2 /
C     DATA IMACH( 5) /   24 /
C     DATA IMACH( 6) / -127 /
C     DATA IMACH( 7) /  127 /
C     DATA IMACH( 8) /   56 /
C     DATA IMACH( 9) / -127 /
C     DATA IMACH(10) /  127 /
C
      ipmpar = imach(i)
      RETURN

      END
