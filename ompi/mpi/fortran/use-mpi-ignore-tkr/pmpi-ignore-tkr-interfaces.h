! -*- fortran -*-
!
! Copyright (c) 2020      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$

#define MPI_Abort PMPI_Abort
#define MPI_Accumulate PMPI_Accumulate
#define MPI_Add_error_class PMPI_Add_error_class
#define MPI_Add_error_code PMPI_Add_error_code
#define MPI_Add_error_string PMPI_Add_error_string
#define MPI_Aint_add PMPI_Aint_add
#define MPI_Aint_diff PMPI_Aint_diff
#define MPI_Allgather PMPI_Allgather
#define MPI_Allgather_init PMPI_Allgather_init
#define MPI_Allgatherv PMPI_Allgatherv
#define MPI_Allgatherv_init PMPI_Allgatherv_init
#define MPI_Alloc_mem PMPI_Alloc_mem
#define MPI_Alloc_mem_cptr PMPI_Alloc_mem_cptr
#define MPI_Allreduce PMPI_Allreduce
#define MPI_Allreduce_init PMPI_Allreduce_init
#define MPI_Alltoall PMPI_Alltoall
#define MPI_Alltoall_init PMPI_Alltoall_init
#define MPI_Alltoallv PMPI_Alltoallv
#define MPI_Alltoallv_init PMPI_Alltoallv_init
#define MPI_Alltoallw PMPI_Alltoallw
#define MPI_Alltoallw_init PMPI_Alltoallw_init
#define MPI_Barrier PMPI_Barrier
#define MPI_Barrier_init PMPI_Barrier_init
#define MPI_Bcast PPMPI_Bcast
#define MPI_Bcast_init PPMPI_Bcast_init
#define MPI_Bsend PMPI_Bsend
#define MPI_Bsend_init PMPI_Bsend_init
#define MPI_Buffer_attach PMPI_Buffer_attach
#define MPI_Buffer_detach PMPI_Buffer_detach
#define MPI_Cancel PMPI_Cancel
#define MPI_Cart_coords PMPI_Cart_coords
#define MPI_Cart_create PMPI_Cart_create
#define MPI_Cart_get PMPI_Cart_get
#define MPI_Cart_map PMPI_Cart_map
#define MPI_Cart_rank PMPI_Cart_rank
#define MPI_Cart_shift PMPI_Cart_shift
#define MPI_Cart_sub PMPI_Cart_sub
#define MPI_Cartdim_get PMPI_Cartdim_get
#define MPI_Close_port PMPI_Close_port
#define MPI_Comm_accept PMPI_Comm_accept
#define MPI_Comm_call_errhandler PMPI_Comm_call_errhandler
#define MPI_Comm_compare PMPI_Comm_compare
#define MPI_Comm_connect PMPI_Comm_connect
#define MPI_Comm_create PMPI_Comm_create
#define MPI_Comm_create_errhandler PMPI_Comm_create_errhandler
#define MPI_Comm_create_group PMPI_Comm_create_group
#define MPI_Comm_create_keyval PMPI_Comm_create_keyval
#define MPI_Comm_delete_attr PMPI_Comm_delete_attr
#define MPI_Comm_disconnect PMPI_Comm_disconnect
#define MPI_Comm_dup PMPI_Comm_dup
#define MPI_Comm_dup_with_info PMPI_Comm_dup_with_info
#define MPI_Comm_free PMPI_Comm_free
#define MPI_Comm_free_keyval PMPI_Comm_free_keyval
#define MPI_Comm_get_attr PMPI_Comm_get_attr
#define MPI_Comm_get_errhandler PMPI_Comm_get_errhandler
#define MPI_Comm_get_info PMPI_Comm_get_info
#define MPI_Comm_get_name PMPI_Comm_get_name
#define MPI_Comm_get_parent PMPI_Comm_get_parent
#define MPI_Comm_group PMPI_Comm_group
#define MPI_Comm_idup PMPI_Comm_idup
#define MPI_Comm_idup_with_info PMPI_Comm_idup_with_info
#define MPI_Comm_join PMPI_Comm_join
#define MPI_Comm_rank PMPI_Comm_rank
#define MPI_Comm_remote_group PMPI_Comm_remote_group
#define MPI_Comm_remote_size PMPI_Comm_remote_size
#define MPI_Comm_set_attr PMPI_Comm_set_attr
#define MPI_Comm_set_errhandler PMPI_Comm_set_errhandler
#define MPI_Comm_set_info PMPI_Comm_set_info
#define MPI_Comm_set_name PMPI_Comm_set_name
#define MPI_Comm_size PMPI_Comm_size
#define MPI_Comm_spawn PMPI_Comm_spawn
#define MPI_Comm_spawn_multiple PMPI_Comm_spawn_multiple
#define MPI_Comm_split PMPI_Comm_split
#define MPI_Comm_split_type PMPI_Comm_split_type
#define MPI_Comm_test_inter PMPI_Comm_test_inter
#define MPI_Compare_and_swap PMPI_Compare_and_swap
#define MPI_Dims_create PMPI_Dims_create
#define MPI_Dist_graph_create PMPI_Dist_graph_create
#define MPI_Dist_graph_create_adjacent PMPI_Dist_graph_create_adjacent
#define MPI_Dist_graph_neighbors PMPI_Dist_graph_neighbors
#define MPI_Dist_graph_neighbors_count PMPI_Dist_graph_neighbors_count
#define MPI_Errhandler_free PMPI_Errhandler_free
#define MPI_Error_class PMPI_Error_class
#define MPI_Error_string PMPI_Error_string
#define MPI_Exscan PMPI_Exscan
#define MPI_Exscan_init PMPI_Exscan_init
#define MPI_F_sync_reg PMPI_F_sync_reg
#define MPI_Fetch_and_op PMPI_Fetch_and_op
#define MPI_Finalize PMPI_Finalize
#define MPI_Finalized PMPI_Finalized
#define MPI_Free_mem PMPI_Free_mem
#define MPI_Gather PMPI_Gather
#define MPI_Gather_init PMPI_Gather_init
#define MPI_Gatherv PMPI_Gatherv
#define MPI_Gatherv_init PMPI_Gatherv_init
#define MPI_Get PMPI_Get
#define MPI_Get_accumulate PMPI_Get_accumulate
#define MPI_Get_address PMPI_Get_address
#define MPI_Get_count PMPI_Get_count
#define MPI_Get_elements PMPI_Get_elements
#define MPI_Get_elements_x PMPI_Get_elements_x
#define MPI_Get_library_version PMPI_Get_library_version
#define MPI_Get_processor_name PMPI_Get_processor_name
#define MPI_Get_version PMPI_Get_version
#define MPI_Graph_create PMPI_Graph_create
#define MPI_Graph_get PMPI_Graph_get
#define MPI_Graph_map PMPI_Graph_map
#define MPI_Graph_neighbors PMPI_Graph_neighbors
#define MPI_Graph_neighbors_count PMPI_Graph_neighbors_count
#define MPI_Graphdims_get PMPI_Graphdims_get
#define MPI_Grequest_complete PMPI_Grequest_complete
#define MPI_Grequest_start PMPI_Grequest_start
#define MPI_Group_compare PMPI_Group_compare
#define MPI_Group_difference PMPI_Group_difference
#define MPI_Group_excl PMPI_Group_excl
#define MPI_Group_free PMPI_Group_free
#define MPI_Group_incl PMPI_Group_incl
#define MPI_Group_intersection PMPI_Group_intersection
#define MPI_Group_range_excl PMPI_Group_range_excl
#define MPI_Group_range_incl PMPI_Group_range_incl
#define MPI_Group_rank PMPI_Group_rank
#define MPI_Group_size PMPI_Group_size
#define MPI_Group_translate_ranks PMPI_Group_translate_ranks
#define MPI_Group_union PMPI_Group_union
#define MPI_Iallgather PMPI_Iallgather
#define MPI_Iallgatherv PMPI_Iallgatherv
#define MPI_Iallreduce PMPI_Iallreduce
#define MPI_Ialltoall PMPI_Ialltoall
#define MPI_Ialltoallv PMPI_Ialltoallv
#define MPI_Ialltoallw PMPI_Ialltoallw
#define MPI_Ibarrier PMPI_Ibarrier
#define MPI_Ibcast PMPI_Ibcast
#define MPI_Ibsend PMPI_Ibsend
#define MPI_Iexscan PMPI_Iexscan
#define MPI_Igather PMPI_Igather
#define MPI_Igatherv PMPI_Igatherv
#define MPI_Improbe PMPI_Improbe
#define MPI_Imrecv PMPI_Imrecv
#define MPI_Ineighbor_allgather PMPI_Ineighbor_allgather
#define MPI_Ineighbor_allgatherv PMPI_Ineighbor_allgatherv
#define MPI_Ineighbor_alltoall PMPI_Ineighbor_alltoall
#define MPI_Ineighbor_alltoallv PMPI_Ineighbor_alltoallv
#define MPI_Ineighbor_alltoallw PMPI_Ineighbor_alltoallw
#define MPI_Info_create PMPI_Info_create
#define MPI_Info_delete PMPI_Info_delete
#define MPI_Info_dup PMPI_Info_dup
#define MPI_Info_free PMPI_Info_free
#define MPI_Info_get PMPI_Info_get
#define MPI_Info_get_nkeys PMPI_Info_get_nkeys
#define MPI_Info_get_nthkey PMPI_Info_get_nthkey
#define MPI_Info_get_string PMPI_Info_get_string
#define MPI_Info_get_valuelen PMPI_Info_get_valuelen
#define MPI_Info_set PMPI_Info_set
#define MPI_Init PMPI_Init
#define MPI_Init_thread PMPI_Init_thread
#define MPI_Initialized PMPI_Initialized
#define MPI_Intercomm_create PMPI_Intercomm_create
#define MPI_Intercomm_merge PMPI_Intercomm_merge
#define MPI_Iprobe PMPI_Iprobe
#define MPI_Irecv PMPI_Irecv
#define MPI_Ireduce PMPI_Ireduce
#define MPI_Ireduce_scatter PMPI_Ireduce_scatter
#define MPI_Ireduce_scatter_block PMPI_Ireduce_scatter_block
#define MPI_Irsend PMPI_Irsend
#define MPI_Is_thread_main PMPI_Is_thread_main
#define MPI_Iscan PMPI_Iscan
#define MPI_Iscatter PMPI_Iscatter
#define MPI_Iscatterv PMPI_Iscatterv
#define MPI_Isend PMPI_Isend
#define MPI_Issend PMPI_Issend
#define MPI_Lookup_name PMPI_Lookup_name
#define MPI_Mprobe PMPI_Mprobe
#define MPI_Mrecv PMPI_Mrecv
#define MPI_Neighbor_allgather PMPI_Neighbor_allgather
#define MPI_Neighbor_allgather_init PMPI_Neighbor_allgather_init
#define MPI_Neighbor_allgatherv PMPI_Neighbor_allgatherv
#define MPI_Neighbor_allgatherv_init PMPI_Neighbor_allgatherv_init
#define MPI_Neighbor_alltoall PMPI_Neighbor_alltoall
#define MPI_Neighbor_alltoall_init PMPI_Neighbor_alltoall_init
#define MPI_Neighbor_alltoallv PMPI_Neighbor_alltoallv
#define MPI_Neighbor_alltoallv_init PMPI_Neighbor_alltoallv_init
#define MPI_Neighbor_alltoallw PMPI_Neighbor_alltoallw
#define MPI_Neighbor_alltoallw_init PMPI_Neighbor_alltoallw_init
#define MPI_Op_commutative PMPI_Op_commutative
#define MPI_Op_create PMPI_Op_create
#define MPI_Op_free PMPI_Op_free
#define MPI_Open_port PMPI_Open_port
#define MPI_Pack PMPI_Pack
#define MPI_Pack_external PMPI_Pack_external
#define MPI_Pack_external_size PMPI_Pack_external_size
#define MPI_Pack_size PMPI_Pack_size
#define MPI_Parrived PMPI_Parrived
#define MPI_Pcontrol PMPI_Pcontrol
#define MPI_Pready PMPI_Pready
#define MPI_Pready_list PMPI_Pready_list
#define MPI_Pready_range PMPI_Pready_range
#define MPI_Precv_init PMPI_Precv_init
#define MPI_Probe PMPI_Probe
#define MPI_Psend_init PMPI_Psend_init
#define MPI_Publish_name PMPI_Publish_name
#define MPI_Put PMPI_Put
#define MPI_Query_thread PMPI_Query_thread
#define MPI_Raccumulate PMPI_Raccumulate
#define MPI_Recv PMPI_Recv
#define MPI_Recv_init PMPI_Recv_init
#define MPI_Reduce PMPI_Reduce
#define MPI_Reduce_init PMPI_Reduce_init
#define MPI_Reduce_local PMPI_Reduce_local
#define MPI_Reduce_scatter PMPI_Reduce_scatter
#define MPI_Reduce_scatter_init PMPI_Reduce_scatter_init
#define MPI_Reduce_scatter_block PMPI_Reduce_scatter_block
#define MPI_Reduce_scatter_block_init PMPI_Reduce_scatter_block_init
#define MPI_Register_datarep PMPI_Register_datarep
#define MPI_Request_free PMPI_Request_free
#define MPI_Request_get_status PMPI_Request_get_status
#define MPI_Rget PMPI_Rget
#define MPI_Rget_accumulate PMPI_Rget_accumulate
#define MPI_Rput PMPI_Rput
#define MPI_Rsend PMPI_Rsend
#define MPI_Rsend_init PMPI_Rsend_init
#define MPI_Scan PMPI_Scan
#define MPI_Scan_init PMPI_Scan_init
#define MPI_Scatter PMPI_Scatter
#define MPI_Scatter_init PMPI_Scatter_init
#define MPI_Scatterv PMPI_Scatterv
#define MPI_Scatterv_init PMPI_Scatterv_init
#define MPI_Send PMPI_Send
#define MPI_Send_init PMPI_Send_init
#define MPI_Sendrecv PMPI_Sendrecv
#define MPI_Sendrecv_replace PMPI_Sendrecv_replace
#define MPI_Ssend PMPI_Ssend
#define MPI_Ssend_init PMPI_Ssend_init
#define MPI_Start PMPI_Start
#define MPI_Startall PMPI_Startall
#define MPI_Status_f2f08 PMPI_Status_f2f08
#define MPI_Status_f082f PMPI_Status_f082f
#define MPI_Status_set_cancelled PMPI_Status_set_cancelled
#define MPI_Status_set_elements PMPI_Status_set_elements
#define MPI_Status_set_elements_x PMPI_Status_set_elements_x
#define MPI_Test PMPI_Test
#define MPI_Test_cancelled PMPI_Test_cancelled
#define MPI_Testall PMPI_Testall
#define MPI_Testany PMPI_Testany
#define MPI_Testsome PMPI_Testsome
#define MPI_Topo_test PMPI_Topo_test
#define MPI_Type_commit PMPI_Type_commit
#define MPI_Type_contiguous PMPI_Type_contiguous
#define MPI_Type_create_darray PMPI_Type_create_darray
#define MPI_Type_create_f90_complex PMPI_Type_create_f90_complex
#define MPI_Type_create_f90_integer PMPI_Type_create_f90_integer
#define MPI_Type_create_f90_real PMPI_Type_create_f90_real
#define MPI_Type_create_hindexed PMPI_Type_create_hindexed
#define MPI_Type_create_hindexed_block PMPI_Type_create_hindexed_block
#define MPI_Type_create_hvector PMPI_Type_create_hvector
#define MPI_Type_create_indexed_block PMPI_Type_create_indexed_block
#define MPI_Type_create_keyval PMPI_Type_create_keyval
#define MPI_Type_create_resized PMPI_Type_create_resized
#define MPI_Type_create_struct PMPI_Type_create_struct
#define MPI_Type_create_subarray PMPI_Type_create_subarray
#define MPI_Type_delete_attr PMPI_Type_delete_attr
#define MPI_Type_dup PMPI_Type_dup
#define MPI_Type_free PMPI_Type_free
#define MPI_Type_free_keyval PMPI_Type_free_keyval
#define MPI_Type_get_attr PMPI_Type_get_attr
#define MPI_Type_get_contents PMPI_Type_get_contents
#define MPI_Type_get_envelope PMPI_Type_get_envelope
#define MPI_Type_get_extent PMPI_Type_get_extent
#define MPI_Type_get_extent_x PMPI_Type_get_extent_x
#define MPI_Type_get_name PMPI_Type_get_name
#define MPI_Type_get_true_extent PMPI_Type_get_true_extent
#define MPI_Type_get_true_extent_x PMPI_Type_get_true_extent_x
#define MPI_Type_indexed PMPI_Type_indexed
#define MPI_Type_match_size PMPI_Type_match_size
#define MPI_Type_set_attr PMPI_Type_set_attr
#define MPI_Type_set_name PMPI_Type_set_name
#define MPI_Type_size PMPI_Type_size
#define MPI_Type_size_x PMPI_Type_size_x
#define MPI_Type_vector PMPI_Type_vector
#define MPI_Unpack PMPI_Unpack
#define MPI_Unpack_external PMPI_Unpack_external
#define MPI_Unpublish_name PMPI_Unpublish_name
#define MPI_Wait PMPI_Wait
#define MPI_Waitall PMPI_Waitall
#define MPI_Waitany PMPI_Waitany
#define MPI_Waitsome PMPI_Waitsome
#define MPI_Win_allocate PMPI_Win_allocate
#define MPI_Win_allocate_cptr PMPI_Win_allocate_cptr
#define MPI_Win_allocate_shared PMPI_Win_allocate_shared
#define MPI_Win_allocate_shared_cptr PMPI_Win_allocate_shared_cptr
#define MPI_Win_attach PMPI_Win_attach
#define MPI_Win_call_errhandler PMPI_Win_call_errhandler
#define MPI_Win_complete PMPI_Win_complete
#define MPI_Win_create PMPI_Win_create
#define MPI_Win_create_dynamic PMPI_Win_create_dynamic
#define MPI_Win_create_errhandler PMPI_Win_create_errhandler
#define MPI_Win_create_keyval PMPI_Win_create_keyval
#define MPI_Win_delete_attr PMPI_Win_delete_attr
#define MPI_Win_detach PMPI_Win_detach
#define MPI_Win_fence PMPI_Win_fence
#define MPI_Win_flush PMPI_Win_flush
#define MPI_Win_flush_all PMPI_Win_flush_all
#define MPI_Win_flush_local PMPI_Win_flush_local
#define MPI_Win_flush_local_all PMPI_Win_flush_local_all
#define MPI_Win_free PMPI_Win_free
#define MPI_Win_free_keyval PMPI_Win_free_keyval
#define MPI_Win_get_attr PMPI_Win_get_attr
#define MPI_Win_get_errhandler PMPI_Win_get_errhandler
#define MPI_Win_get_group PMPI_Win_get_group
#define MPI_Win_get_info PMPI_Win_get_info
#define MPI_Win_get_name PMPI_Win_get_name
#define MPI_Win_lock PMPI_Win_lock
#define MPI_Win_lock_all PMPI_Win_lock_all
#define MPI_Win_post PMPI_Win_post
#define MPI_Win_set_attr PMPI_Win_set_attr
#define MPI_Win_set_errhandler PMPI_Win_set_errhandler
#define MPI_Win_set_info PMPI_Win_set_info
#define MPI_Win_set_name PMPI_Win_set_name
#define MPI_Win_shared_query PMPI_Win_shared_query
#define MPI_Win_shared_query_cptr PMPI_Win_shared_query_cptr
#define MPI_Win_start PMPI_Win_start
#define MPI_Win_sync PMPI_Win_sync
#define MPI_Win_test PMPI_Win_test
#define MPI_Win_unlock PMPI_Win_unlock
#define MPI_Win_unlock_all PMPI_Win_unlock_all
#define MPI_Win_wait PMPI_Win_wait
#define MPI_Wtick PMPI_Wtick
#define MPI_Wtime PMPI_Wtime

#include "ompi/mpi/fortran/use-mpi-ignore-tkr/mpi-ignore-tkr-interfaces.h"
