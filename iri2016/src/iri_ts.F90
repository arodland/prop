program basictest
use, intrinsic:: iso_fortran_env, only: stderr=>error_unit, stdout=>output_unit
implicit none

logical :: jf(50)
integer, parameter :: jmag = 0
integer :: iyyyy, mmdd, Nalt, yyyymod, mmddmod
real :: glat, glon, dhour, hourmod
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
character(500) :: line

integer :: i, io, noff
integer :: offs(121)

jf = .true.
jf(4:6) = .false.
jf(21:23) = .false.
jf(28:30) = .false.
jf(33:35) = .false.
jf(39) = .false.
jf(40) = .false.
jf(47) = .false.

do

  read(*,*, end=42) ymdhms(1), ymdhms(2), ymdhms(3), ymdhms(4), ymdhms(5), ymdhms(6), &
    glat, glon

  read(*, '(A)', end=42) line

  do i=1,120
    read(line, *, iostat=io) offs(1:i)
    if (io==-1) exit
  enddo
  offs(i) = 120
  noff = i

  iyyyy = ymdhms(1)
  mmdd = ymdhms(2) * 100 + ymdhms(3)
  dhour = ymdhms(4) + ymdhms(5) / 60. + ymdhms(6) / 3600.

  datadir1 = datadir
  call read_ig_rz
  call readapf107

  do i=1,noff
    hourmod = dhour + offs(i)/60.
    mmddmod = mmdd
    yyyymod = iyyyy

    if (hourmod.gt.24.) then
      hourmod = hourmod - 24.
      mmddmod = mmddmod + 1
      if (mmddmod.gt.1231) then
        mmddmod = 101
        yyyymod = yyyymod + 1
      endif
    endif

    call IRI_SUB(JF,JMAG,glat,glon,yyyymod,mmddmod,hourmod+25., &
      0, 0, 1, &
      OUTF,OARR, datadir)
  
    write(stdout, '(3ES16.8)', advance="no") oarr(91), oarr(36) * oarr(91), oarr(2)
  enddo
  write(stdout,*) ""

enddo
42 continue
end program

