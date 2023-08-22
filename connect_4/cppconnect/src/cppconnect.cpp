#include "Player.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <string>


NB_MODULE(cppconnect,m) {
    namespace nb = nanobind;
    nb::class_<Self_player> (m,"Self_player")
            .def(nb::init<const std::string&>())
            .def("load_model",&Self_player::load_model)
            .def("play_game",&Self_player::play_game);
    m.def("evaluate",&evaluate);
}