#include "Player.h"
#include <nanobind/nanobind.h>
#include <string>

NB_MODULE(cppconnect,m) {
    namespace nb = nanobind;
    nb::class_<Self_player> (m,"Self_player")
            .def(nb::init<std::string&&>())
            .def("load_model",&Self_player::load_model)
            .def("play_game",&Self_player::play_game);
    m.def("evaluate",&evaluator);
    m.def("evaluate_tournament",&evaluate_tournament);
}