program basictest
use, intrinsic:: iso_fortran_env, only: stderr=>error_unit, stdout=>output_unit
implicit none

logical :: jf(50)
integer, parameter :: jmag = 0
integer :: iyyyy, mmdd, Nalt
real :: glat, glon, dhour
integer :: ymdhms(6)
real:: alt_km_range(3)
real::  TECtotal, TECtop, TECbot

#ifndef BIN_DIR
#define BIN_DIR '.'
#endif
character(*), parameter :: datadir = BIN_DIR // '/iri2020/data'
character(256) :: datadir1
common /folders/ datadir1


real :: oarr(100), outf(20,1000)
real, allocatable :: altkm(:)
character(80) :: argv
integer :: i

jf = .true.
jf(4:6) = .false.
jf(21:23) = .false.
jf(26) = .false.
jf(28:30) = .false.
jf(33:35) = .false.
jf(39) = .false.
jf(47) = .false.

! --- command line input
if (command_argument_count() < 11) then
  write(stderr,*) 'need input parameters: year month day hour minute second glat glon min_alt_km max_alt_km step_alt_km'
  stop 1
endif

do i=1,6
  call get_command_argument(i,argv)
  read(argv,*) ymdhms(i)
enddo

call get_command_argument(7, argv)
read(argv,*) glat

call get_command_argument(8, argv)
read(argv,*) glon

do i = 1,3
  call get_command_argument(8+i, argv)
  read(argv,*) alt_km_range(i)
enddo

! --- parse
Nalt = int((alt_km_range(2) - alt_km_range(1)) / alt_km_range(3)) + 1
allocate(altkm(Nalt))


altkm(1) = alt_km_range(1)
do i = 2,Nalt
  altkm(i) = altkm(i-1) + alt_km_range(3) 
enddo

iyyyy = ymdhms(1)
mmdd = ymdhms(2) * 100 + ymdhms(3)
dhour = ymdhms(4) + ymdhms(5) / 60. + ymdhms(6) / 3600.

datadir1 = datadir
call read_ig_rz
call readapf107

call IRI_SUB(JF,JMAG,glat,glon,IYYYY,MMDD,DHOUR+25., &
     alt_km_range(1), alt_km_range(2), alt_km_range(3), &
     OUTF,OARR, datadir)
 
! --- for TEC
call iri_tec(alt_km_range(1), alt_km_range(2), 2,&
             TECtotal, TECtop, TECbot)
oarr(37) = TECtotal  ! tec-units (10^16 m^-2)
oarr(38) = TECtop ! % of total

!print '(A,ES10.3,A,F5.1,A)','NmF2 ',oarr(1),' [m^-3]     hmF2 ',oarr(2),' [km] '
!print '(A,F10.3,A,I3,A,F10.3)','F10.7 ',oarr(41), ' Ap ',int(oarr(51)),' B0 ',oarr(10)

!print *,'Altitude    Ne    O2+'
do i = 1,Nalt
  write(stdout, '(F10.3, 11ES16.8)') altkm(i), outf(:11,i)
enddo


write(stdout, '(/,100ES16.8)') oarr

end program

