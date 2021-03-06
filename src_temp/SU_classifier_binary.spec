# -*- mode: python -*-

from distutils.sysconfig import get_python_lib
from os import path 
skimage_plugins = Tree(
    path.join(get_python_lib(), "skimage","io","_plugins"), 
    prefix=path.join("skimage","io","_plugins"),
    )

block_cipher = None


a = Analysis(['SU_classifier_binary.py'],
             pathex=['d:\\Projects\\WaterScope'],
             binaries=[],
             datas=[],
             hiddenimports=['cython', 'sklearn', 'sklearn.neighbors.typedefs','pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
	  skimage_plugins,
          name='SU_classifier_binary',
          debug=False,
          strip=False,
          upx=True,
          console=True )
