program basictest
use, intrinsic:: iso_fortran_env, only: stderr=>error_unit, stdout=>output_unit
implicit none

logical :: jf(50)
integer, parameter :: jmag = 0
real :: glat, glon, dhour
integer :: qtime(6)
integer :: tov(5)
integer :: tovdate, tovhhmm, otovdate, otovhhmm
integer :: qdate
integer :: iyyyy, mmdd
real :: tovhour, irtamhour
real :: ff2(1064), fh2(1064), fb0(1064), fb1(1064)
real :: modip
real :: fof2rt, hmf2rt

real, external :: FOUT1

#ifndef BIN_DIR
#define BIN_DIR '.'
#endif
character(*), parameter :: datadir = BIN_DIR // '/iri2020/data'
character(256) :: datadir1
common /folders/ datadir1

real :: oarr(100), outf(20,1000)

jf = .true.
jf(4:6) = .false.
jf(21:23) = .false.
jf(28:30) = .false.
jf(33:36) = .false.
jf(39) = .false.
jf(40) = .false.
jf(47) = .false.

datadir1 = datadir
call read_ig_rz
call readapf107

otovdate = -1
otovhhmm = -1

do
  read(*,*, end=42) glat, glon, & 
    tov(1), tov(2), tov(3), tov(4), tov(5), &
    qtime(1), qtime(2), qtime(3), qtime(4), qtime(5), qtime(6)

  tovdate = tov(1)*10000 + tov(2)*100 + tov(3)
  tovhhmm = tov(4)*100 + tov(5)
  tovhour = tov(4) + tov(5) / 60.

  iyyyy = qtime(1)
  mmdd = qtime(2) * 100 + qtime(3)
  qdate = iyyyy*10000 + mmdd

  dhour = qtime(4) + qtime(5) / 60. + qtime(6) / 3600.
  if(qdate.gt.tovdate) then
      irtamhour = dhour + 24.
  else
      irtamhour = dhour
  endif

  if ((tovdate.ne.otovdate).or.(tovhhmm.ne.otovhhmm)) then
      call READIRTAMCOF(0, tovdate, tovhhmm, 1064, ff2)
      call READIRTAMCOF(1, tovdate, tovhhmm, 1064, fh2)
  endif

  otovdate = tovdate
  otovhhmm = tovhhmm

  call IRI_SUB(JF,JMAG,glat,glon,IYYYY,MMDD,DHOUR+25., &
      0, 0, 1, &
      OUTF, OARR, datadir)

  modip = OARR(27)

  fof2rt = FOUT1(modip, glat, glon, irtamhour, tovhour, ff2)
  hmf2rt = FOUT1(modip, glat, glon, irtamhour, tovhour, fh2)

  write(stdout, '(2ES16.8)') fof2rt, hmf2rt
enddo
42 continue
end program
