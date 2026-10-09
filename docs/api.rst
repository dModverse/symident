API reference
=============

.. currentmodule:: symident

Model and analyses
------------------

.. autoclass:: Model

.. autofunction:: detect

.. autofunction:: reduce

Results
-------

.. autoclass:: Symmetries
   :members: summary, reduce, to_dict

.. autoclass:: Symmetry
   :members: sympy

.. autoclass:: Reduction
   :members: summary, to_dict

Settings and programs
---------------------

.. autofunction:: reconst_control

.. autofunction:: install_msolve

.. autofunction:: symident.install.find_msolve

.. autofunction:: symident.install.cache_dir

.. autoexception:: symident.sym.msolve.MsolveMissingError

JSON interface
--------------

.. automodule:: symident.rjson
   :members: detect_json, reduce_json
