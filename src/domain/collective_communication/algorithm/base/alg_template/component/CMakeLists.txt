set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/reducer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sender.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)