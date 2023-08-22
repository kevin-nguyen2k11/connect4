#ifndef C___CONNECT_4_GAME_CONTROLLER_H
#define C___CONNECT_4_GAME_CONTROLLER_H

#include "Config.h"
#include "Ndarray.h"
#include <cppflow/cppflow.h>
#include <vector>
#include <array>

class Game_controller {
public:
    [[nodiscard]]
    Ndarray<float> get_state() const;

    [[nodiscard]]
    std::array<bool, Config::width> get_legal_moves() const;

    void make_move(int x, std::vector<int>* move_history = nullptr);

    bool is_over{false};
    int num_moves{0};
    int win_state{};

private:
    void is_game_done(int x, int y);

    Ndarray<float> board{{Config::height,Config::width,2}};
    int current_player{1};
};

#endif //C___CONNECT_4_GAME_CONTROLLER_H
