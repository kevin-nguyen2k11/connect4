#include "Game_controller.h"
#include "Config.h"
#include <cppflow/cppflow.h>
#include <array>
#include <algorithm>

Ndarray<float> Game_controller::get_state() const
{
    if (current_player) {
        Ndarray<float> reversed{{Config::height,Config::width,2}};
        for (int i{0}; i<Config::height; ++i) {
            for (int j{0}; j<Config::width; ++j) {
                for (int k{0}; k<2; ++k) {
                    reversed[{i,j,k}] = board[{i,j,2-k-1}];
                }
            }
        }
        return reversed;
    }
    return board;
}

std::array<bool, Config::width> Game_controller::get_legal_moves() const
{
    std::array<bool, Config::width> legal_moves;
    for (int i{0}; i<Config::width; ++i) {
        legal_moves[i] = !(board[{Config::height-1,i,0}] || board[{Config::height-1,i,1}]);
    }
    return legal_moves;
}

void Game_controller::make_move(int x, std::vector<int>* move_history)
{
    current_player ^= 1;
    const int y = [&] {
        int y{0};
        for (int i{0}; i<Config::height; ++i) {
            if (!(board[{i,x,0}] || board[{i,x,1}])) {
                y = i;
                break;
            }
        }
        return y;
    }();

    if (move_history) {
        move_history->insert(move_history->end(),{x,y});
    }
    board[{y,x,current_player}] = 1;
    num_moves += 1;
    is_game_done(x,y);
}

void Game_controller::is_game_done(int x, int y)
{
    std::array<int, 4> sums{1,1,1,1};
    std::array<bool, 8> stop{};
    int not_done{8};
    for (int i{1}; not_done; ++i) {
        for (int j{0}; j<8; ++j) {
            if (!stop[j]) {
                int add_x = (j%2) ? -i*Config::dirs[j/2][0] : i*Config::dirs[j/2][0];
                int add_y = (j%2) ? -i*Config::dirs[j/2][1] : i*Config::dirs[j/2][1];
                if (board[{y+add_y,x+add_x,current_player}]) {
                    sums[j/2] += 1;
                } else {
                    stop[j] = true;
                    not_done -= 1;
                }
            }
        }
    }
    const int state{std::any_of(sums.begin(),sums.end(),[](int i){return i>=Config::win_length;})};
    if (state || num_moves==Config::max_moves) {
        win_state = state;
        is_over = true;
    }
}