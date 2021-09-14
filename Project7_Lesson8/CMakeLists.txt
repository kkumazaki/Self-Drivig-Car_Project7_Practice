cmake_minimum_required(VERSION 3.0.0)
project(Practice_Lesson8 VERSION 0.1.0)

include(CTest)
enable_testing()

#ファイル検索:リストアップ ターケットファイルリストの生成 20210904
file(GLOB Target_Files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.cpp)
#実行ファイルの生成 20210904
add_executable(main ${Target_Files})
#add_executable(Practice_Lesson8 main.cpp) #20210904

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
