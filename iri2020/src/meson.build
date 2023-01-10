project('iri2020', 'fortran', default_options : ['default_library=static'])

add_project_arguments('-O3', '-ffast-math', '-march=znver3', language: 'fortran')

iri2020= library('iri2020', 'irisub.for', 'irifun.for', 'iritec.for', 'iridreg.for', 'igrf.for', 'cira.for', 'iriflip.for', 'rocdrift.for',
  fortran_args: '-w')

driver = executable('iri2020_driver', 'iri2020_driver.F90',
  link_with: iri2020,
  fortran_args: '-DBIN_DIR='+ '"' + meson.build_root() + '"')
  
map = executable('irimap', 'irimap.F90',
  link_with: iri2020,
  fortran_args: '-DBIN_DIR='+ '"' + meson.build_root() + '"')

ts = executable('iri_ts', 'iri_ts.F90',
  link_with: iri2020,
  fortran_args: '-DBIN_DIR='+ '"' + meson.build_root() + '"')

opt = executable('iri_opt', 'iri_opt.F90',
  link_with: iri2020,
  fortran_args: '-DBIN_DIR='+ '"' + meson.build_root() + '"')

testexe = executable('testiri2020', 'test.f90',
  link_with: iri2020)
 
test('IRI2016', testexe)
