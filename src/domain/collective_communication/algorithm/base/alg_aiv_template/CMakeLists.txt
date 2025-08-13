set_source_files_properties(
    aiv_communication.h PROPERTIES LANGUAGE CXX
)
if(BUILD_OPEN_PROJECT)
    set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
    add_custom_target(hccl_aiv_op_ascend910B ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o --cce-aicore-arch=dav-c220-vec
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_custom_target(hccl_ag_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ag_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ag_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ag_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ag_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )


    add_custom_target(hccl_ag_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ag_superkernel_uint8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )
    
    add_custom_target(hccl_ag_superkernel_uint16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ag_superkernel_uint32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )


    add_custom_target(hccl_rs_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_rs_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_rs_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_rs_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_rs_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )


    add_custom_target(hccl_rs_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ar_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ar_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ar_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ar_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ar_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_ar_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

        add_custom_target(hccl_a2a_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_a2a_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_a2a_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_a2a_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_a2a_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )


    add_custom_target(hccl_a2a_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_a2a_superkernel_uint8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )
    
    add_custom_target(hccl_a2a_superkernel_uint16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )

    add_custom_target(hccl_a2a_superkernel_uint32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            
    )
else()
    get_filename_component(CCE_DIR "${CMAKE_LINKER}" PATH)

    set(TILING_GEN_DIR ${CMAKE_BINARY_DIR}/atc/opcompiler/ascendc_compiler/api)

    add_custom_target(hccl_aiv_op_ascend910B
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o --cce-aicore-arch=dav-c220-vec
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

        add_custom_target(hccl_ag_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )


    add_custom_target(hccl_ag_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_uint8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )
    
    add_custom_target(hccl_ag_superkernel_uint16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_uint32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )


    add_custom_target(hccl_rs_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )


    add_custom_target(hccl_rs_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

        add_custom_target(hccl_a2a_superkernel_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )


    add_custom_target(hccl_a2a_superkernel_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_uint8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT8
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )
    
    add_custom_target(hccl_a2a_superkernel_uint16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT16
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_uint32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_UINT32
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS kernel_tiling_headers
    )

endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int8.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_int32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_float16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_float32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_bfloat16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint8.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_uint32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int8.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_int32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_float16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_float32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_bfloat16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint8.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_uint32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int8.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_int32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_float16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_float32.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_bfloat16.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)
