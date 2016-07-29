.. _api_ref:

.. currentmodule:: clustertracking

API reference
=============

Core functionality
------------------
:func:`~clustertracking.find_link` links and segments the images and
:func:`~clustertracking.refine_leastsq` refines the positions by fitting
to sums of model functions.

.. autosummary::
   :toctree: generated/

   find_link
   refine_leastsq


Constraints
-----------

.. autosummary::
    :toctree: generated/

   constraints.dimer
   constraints.trimer
   constraints.tetramer
   constraints.dimer_global


Model image creation
--------------------

.. autosummary::
    :toctree: generated/

   artificial.SimulatedImage
   artificial.CoordinateReader
   artificial.get_single
   artificial.get_dimer
   artificial.get_multiple

Cluster motion analysis
-----------------------

.. autosummary::
    :toctree: generated/

   motion.orientation_df
   motion.diffusion_tensor
   motion.diffusion_tensor_ci
   motion.friction_tensor

Helper functions
----------------

.. autosummary::
    :toctree: generated/

   find.find_clusters
   fitfunc.FitFunctions
   fitfunc.vect_from_params
   fitfunc.vect_to_params
   masks.slice_pad
   masks.slices_multiple
   masks.slice_image
   masks.mask_image
   masks.binary_mask_multiple