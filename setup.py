import os, sys
import pathlib, glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Hack:
# trying to locate from current search locations the folder 
# containing the tensorflow package, as we need tensorflow to provide
# compile/link flags to build the .so
tf_package_search_paths = []
for candidate_path in sys.path:
  print(candidate_path)
  search_paths = []
  for package_path in glob.iglob(os.path.join(candidate_path, '**/tensorflow/'), recursive=True):
    search_paths.append(os.path.dirname(os.path.normpath(package_path)))

  try:
    tf_package_search_paths.append(os.path.commonpath(search_paths))
  except ValueError as e:
    pass  

print(f'Adding extra search paths for locating tensorflow package:\n{tf_package_search_paths}')
tf_package_search_paths = ':'.join(tf_package_search_paths)


class TF_Kernel(Extension):
  def __init__(self, name):
    super().__init__(name, sources=[])

class build_ext_supporting_TF_Kernel(build_ext):

  def get_ext_filename(self, fullname):
    ext = self.ext_map.get(fullname, None)
    if not isinstance(ext, TF_Kernel):
      return super().get_ext_filename(fullname)
    else:
      # Fixing the output suffix of a TF_Kernel to '.so'
      return os.path.join(*fullname.split('.')) + '.so'

  def build_extension(self, ext):
    if not isinstance(ext, TF_Kernel):
      super().build_extension(ext)
    else:
      # TF_Kernel is built by make
      fullname = self.get_ext_fullname(ext.name)
      split_at = fullname.rfind('.')
      if split_at < 0:
        raise ValueError(f'Invalid TF_Kernel name: {fullname}')

      module_package_name = fullname[:split_at]
      module_name = fullname[split_at+1:]

      build_py = self.get_finalized_command('build_py')
      src_dir = build_py.get_package_dir(module_package_name)

      install_dir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent
      install_dir.mkdir(parents=True, exist_ok=True)
    
      command = ['make', '-C', f'{src_dir}',
                  f'TF_PACKAGE_SEARCH_PATHS={tf_package_search_paths}',
                  f'MODULE_NAME={module_name}',
                  f'INSTALL_TO={install_dir.absolute()}',
                  'all',
                  'install',
                  'clean'
                ]

      if self.dry_run:
        print(f'{type(self).__name__} build: {command.join(" ")}')
      else:
        self.spawn(command)
        
setup(
  ext_modules=[TF_Kernel('flash_attention.kernel.flash_attention')],
  cmdclass={
          'build_ext': build_ext_supporting_TF_Kernel,
  }
)

