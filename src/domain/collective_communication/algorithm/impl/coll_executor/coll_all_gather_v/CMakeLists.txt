set(src_list
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_v_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_v_mesh_opbase_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gatherv_mesh_aiv_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gatherv_mesh_aiv_smallcount_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_v_mesh_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_v_ring_for_910_93_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_aligned_all_gather_v_double_ring_for_910_93_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_v_semi_ring_executor.cc
)

target_sources(hccl_alg PRIVATE
  ${src_list}
)

add_subdirectory(310P)