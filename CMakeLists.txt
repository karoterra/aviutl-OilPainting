cmake_minimum_required(VERSION 3.20)
project(OilPainting)

set(LUA_INCLUDE_DIR "vendor/lua/include" CACHE PATH "lua 5.1.4 include dir")
set(LUA_LIB_DIR "vendor/lua" CACHE PATH "lua 5.1.4 dll dir")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "../aviutl" CACHE PATH "..." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DLL_NAME KaroterraOilPainting)

find_package(OpenMP REQUIRED)

add_library(${DLL_NAME} SHARED
    src/main.cpp
    src/resource.h
    src/OilPainting.rc
)
target_include_directories(${DLL_NAME} PRIVATE ${LUA_INCLUDE_DIR})
target_link_directories(${DLL_NAME} PRIVATE ${LUA_LIB_DIR})
target_link_libraries(${DLL_NAME} PRIVATE lua51 OpenMP::OpenMP_CXX)

install(TARGETS ${DLL_NAME}
    RUNTIME DESTINATION script
)
install(FILES
    "script/油絵.anm"
    DESTINATION script
)
