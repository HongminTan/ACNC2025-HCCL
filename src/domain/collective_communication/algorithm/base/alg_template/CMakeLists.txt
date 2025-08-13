set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/recursive_halvingdoubling_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_hierarchical_ring_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_hierarchical_ring_v1_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_bruck_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_base.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)

add_subdirectory(component)
add_subdirectory(temp_alltoall)
add_subdirectory(temp_all_gather)
add_subdirectory(temp_all_reduce)
add_subdirectory(temp_alltoallv)
add_subdirectory(temp_broadcast)
add_subdirectory(temp_gather)
add_subdirectory(temp_reduce)
add_subdirectory(temp_reduce_scatter)
add_subdirectory(temp_scatter)
add_subdirectory(temp_send_recv)
