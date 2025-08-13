set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hccl_one_sided_service.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_one_sided_service.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_one_sided_conn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/one_sided_service_adapt.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)
