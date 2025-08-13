set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_single_rank_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_fast_double_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_aligned_all_reduce_double_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_double_ring_concurrent_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_mid_count_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_oneshot_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_opbase_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_opbase_pipeline_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_graph_pipeline_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_small_count_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_reduce_plus_bcast_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_aiv_smallcount_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_small_count_aiv_rdma_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mid_count_aiv_rdma_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mix_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_ring_zerocopy_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_order_preserved_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_aiv_deter_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_aiv_deter_small_executor.cc

)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(310P)
