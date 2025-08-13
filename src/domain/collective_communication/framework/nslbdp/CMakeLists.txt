set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_nslbdp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_nslb_md5.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)