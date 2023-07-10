set(PYMODULES "pymatchlr")

foreach(module ${PYMODULES})
    pybind11_add_module("${module}" "${module}.cpp")
    target_link_libraries("${module}" 
        PRIVATE pybind11::module 
        PUBLIC ${OTHER_LIBS})
    target_include_directories("${module}" 
        PUBLIC ${PUBLIC_INCLUDE})
endforeach(module ${PYMODULES})
