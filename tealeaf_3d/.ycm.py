import os
import ycm_core

def DirectoryOfThisScript():
  return os.path.dirname( os.path.abspath( __file__ ) )

def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
      return list( flags )
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag

    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )

    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break

      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break

    if new_flag:
      new_flags.append( new_flag )
  return new_flags

def FlagsForFile( filename, **kwargs):

  flags = [
      '-Wall',
      '-Wextra',
      '-Werror',
      '-pedantic',
      '-fexceptions',
      '-lm',
      '-I.',
      '-Ikokkos/core/src/', 
      '-Ikokkos/containers/src/', 
      '-Ikokkos/algorithms/src/',
      '/usr/include',
      '-isystem', '/System/Library/Frameworks/Python.framework/Headers',
      '-isystem', '/usr/local/include',
      '-isystem', '/usr/local/include/eigen3',
      ]

  data = kwargs['client_data']
  filetype = data['&filetype']

  if filetype == 'c' or filetype == 'h':
    flags += ['-xc']
    flags += ['-std=c99']
  elif filetype == 'cpp' or filetype == 'hpp':
    flags += ['-xc++']
    flags += ['-std=c++11']
  else:
    flags = []

  relative_to = DirectoryOfThisScript()
  final_flags = MakeRelativePathsInFlagsAbsolute( flags, relative_to )

  return {
    'flags': final_flags,
    'do_cache': True
  }
