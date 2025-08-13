set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/bcast_halvingdoubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/bcast_recursive_halvingdoubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_star.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_nhr_oneshot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_nhr_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_nb_binary.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_oneshot.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)