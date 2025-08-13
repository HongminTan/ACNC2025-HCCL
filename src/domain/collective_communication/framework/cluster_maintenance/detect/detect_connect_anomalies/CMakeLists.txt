set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_connect_anomalies.cc
)


target_sources(hccl PRIVATE
    ${src_list}
)