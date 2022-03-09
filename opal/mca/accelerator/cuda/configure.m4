# -*- shell-script -*-
#
# Copyright (c) 2011-2013 NVIDIA Corporation.  All rights reserved.
# Copyright (c) 2013      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2022      Amazon.com, Inc. or its affiliates.
#                         All Rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

#
# If CUDA support was requested, then build the CUDA support library.
# This code checks just makes sure the check was done earlier by the
# opal_check_cuda.m4 code.
#

AC_DEFUN([MCA_opal_accelerator_cuda_CONFIG],[
    AC_CONFIG_FILES([opal/mca/accelerator/cuda/Makefile])

    # make sure that CUDA-aware checks have been done
    AC_REQUIRE([OPAL_CHECK_CUDA])

    AS_IF([test "x$CUDA_SUPPORT" = "x1"],
          [$1],
          [$2])

])dnl

