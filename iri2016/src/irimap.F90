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
character(*), parameter :: datadir = BIN_DIR // '/iri2016/data'
character(256) :: datadir1
common /folders/ datadir1

real :: oarr(100), outf(20,1000)
real, allocatable :: altkm(:)
character(80) :: argv
integer :: i

jf = .true.
jf(4:6) = .false.
jf(21:23) = .false.
jf(28:30) = .false.
jf(33:35) = .false.

! --- command line input
if (command_argument_count() < 6) then
  write(stderr,*) 'need input parameters: year month day hour minute second'
  stop 1
endif

do i=1,6
  call get_command_argument(i,argv)
  read(argv,*) ymdhms(i)
enddo

iyyyy = ymdhms(1)
mmdd = ymdhms(2) * 100 + ymdhms(3)
dhour = ymdhms(4) + ymdhms(5) / 60. + ymdhms(6) / 3600.

datadir1 = datadir
call read_ig_rz
call readapf107

do glat=-90,90
  do glon=-180,180
    call IRI_SUB(JF,JMAG,glat,glon,IYYYY,MMDD,DHOUR+25., &
      0, 0, 1, &
      OUTF,OARR, datadir)
 
    ! lat, lon, nmf2, fof2, m(3000), muf(3000), foe
    write(stdout, '(7ES16.8)') glat, glon, oarr(1), oarr(91), oarr(36), oarr(36) * oarr(89), oarr(92)
  enddo
enddo

end program

