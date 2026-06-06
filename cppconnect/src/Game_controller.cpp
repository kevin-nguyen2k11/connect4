#include "Game_controller.h"
#include "Config.h"
#include <vector>
#include <array>
#include <algorithm>
#include <sstream>
#include <iterator>

void Game_controller::get_state(Ndarray<float>& inputs, int thread) const
{
    for (int k{0}; k<2; ++k) {
        const int l = (current_player) ? (1-k) : k;
        for (int i{0}; i<Config::height; ++i) {
            for (int j{0}; j<Config::width; ++j) {
                inputs[{thread,i,j,k}] = board.read({i,j,l});
            }
        }
    }
}

std::vector<bool> Game_controller::get_legal_moves() const
{
    std::vector<bool> legal_moves(Config::width);
    for (int i{0}; i<Config::width; ++i) {
        legal_moves[i] = !(board.read({Config::height-1,i,0}) || board.read({Config::height-1,i,1}));
    }
    return legal_moves;
}

int Game_controller::make_move(int x, std::vector<int>* move_history)
{
    current_player ^= 1;
    const int y = [&] {
        int y{0};
        for (int i{0}; i<Config::height; ++i) {
            if (!(board.read({i,x,0}) || board.read({i,x,1}))) {
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
    return y;
}

int Game_controller::get_result(int x, int y) const
{
    std::array<int, 4> sums{1,1,1,1};
    std::array<bool, 8> stop{};
    int not_done{8};
    for (int i{1}; not_done; ++i) {
        for (int j{0}; j<8; ++j) {
            if (!stop[j]) {
                int add_x = (j%2) ? -i*Config::dirs[j/2][0] : i*Config::dirs[j/2][0];
                int add_y = (j%2) ? -i*Config::dirs[j/2][1] : i*Config::dirs[j/2][1];
                if (board.read({y+add_y,x+add_x,current_player})) {
                    sums[j/2] += 1;
                } else {
                    stop[j] = true;
                    not_done -= 1;
                }
            }
        }
    }
    const int state{std::any_of(sums.begin(),sums.end(),[](int i){return i>=Config::win_length;})};
    return (state || num_moves==Config::max_moves) ? state : 2;
}

void Game_controller::print_board() const
{
    std::vector<std::string> string_board;
    for (int i{0}; i<Config::height; ++i) {
        string_board.emplace_back(" - - - - - - - \n");
        for (int j{0}; j<Config::width; ++j) {
            std::string value;
            if (board.read({i,j,0})==0 && board.read({i,j,1})==0) {
                value = " ";
            }
            else {
                value = (board.read({i,j,0})==1) ? "x" : "o";
            }
            string_board.emplace_back("|" + value);
        }
        string_board.emplace_back("|\n");
    }
    string_board.emplace_back(" - - - - - - - \n");
    std::ostringstream imploded;
    std::copy(string_board.begin(),string_board.end(),std::ostream_iterator<std::string>(imploded, ""));
    std::cout << imploded.str() << std::endl;
}