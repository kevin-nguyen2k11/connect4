#ifndef C___CONNECT_4_GAME_CONTROLLER_H
#define C___CONNECT_4_GAME_CONTROLLER_H

#include "Config.h"
#include "Ndarray.h"
#include <vector>
#include <array>

class Game_controller {
public:
    void get_state(Ndarray<float>& inputs, int thread) const;

    [[nodiscard]]
    std::vector<bool> get_legal_moves() const;

    int make_move(int x, std::vector<int>* move_history = nullptr);

    [[nodiscard]]
    int get_result(int x, int y) const;

    void print_board() const;

    int num_moves{0};

private:

    Ndarray<float> board{{Config::height,Config::width,2}};
    int current_player{1};
};

#endif //C___CONNECT_4_GAME_CONTROLLER_H
