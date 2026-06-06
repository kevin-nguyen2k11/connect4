#include "Game_controller.h"
#include "Config.h"
#include "Ndarray.h"
#include <array>
#include <queue>
#include <vector>
#include <sstream>
#include <iterator>

Ndarray<float> Game_controller::get_state() const
{
    if (current_player) {
        Ndarray<float> temp{board};
        for (int i{0}; i<2; ++i) {
            const int row{pos[i][0]};
            const int col{pos[i][1]};
            temp[{row,col,0}] = i;
            temp[{row,col,1}] = i^1;
        }
        return temp;
    }
    return board;
}

std::vector<int> Game_controller::get_legal_moves()
{
    const int start_row{pos[current_player^1][0]};
    const int start_col{pos[current_player^1][1]};
    const int opp_row{pos[current_player][0]};
    const int opp_col{pos[current_player][1]};
    Ndarray<int> legal_moves{{Config::size,Config::size,4}};
    legal_moves.fill(0);
    std::array<std::array<bool, Config::size>, Config::size> visited{false};
    std::queue<std::vector<int>> node_queue{};
    node_queue.emplace(std::vector<int>{start_row,start_col,0});
    visited[start_row][start_col] = true;
    bool found_opp{false};
    bool within_distance{true};
    int area{0};
    while (!node_queue.empty()) {
        area += 1;
        std::vector cur_pos{node_queue.front()};
        const int row{cur_pos[0]};
        const int col{cur_pos[1]};
        node_queue.pop();
        if (cur_pos[2]==Config::max_steps) {
            within_distance = false;
        }
        if (row==opp_row && col==opp_col) {
            found_opp = true;
            continue;
        }
        if (!within_distance && found_opp) {
            break;
        }
        for (int i{0}; i<4; ++i) {
            if (!board.read({row,col,i+2}) && !legal_moves.read({row,col,i})) {
                const int row_new{row + Config::dirs[i][0]};
                const int col_new{col + Config::dirs[i][1]};
                if (within_distance) {
                    legal_moves[{row,col,i}] = 1;
                    if ((cur_pos[2]+1 != Config::max_steps) && !(row_new==opp_row && col_new==opp_col)) {
                        legal_moves[{row_new,col_new,(i+2)%4}] = 1;
                    }
                }
                if (!visited[row_new][col_new]) {
                    node_queue.emplace(std::vector<int>{row_new,col_new,cur_pos[2]+1});
                    visited[row_new][col_new] = true;
                }
            }
        }
    }
    if (!found_opp) {
        std::array<std::array<bool, Config::size>, Config::size> opp_visited{false};
        node_queue.emplace(std::vector<int>{opp_row,opp_col});
        opp_visited[opp_row][opp_col] = true;
        int opp_area{0};
        while (!node_queue.empty()) {
            opp_area += 1;
            std::vector cur_pos{node_queue.front()};
            const int row{cur_pos[0]};
            const int col{cur_pos[1]};
            node_queue.pop();
            for (int i{0}; i<4; ++i) {
                if (!board.read({row,col,i+2})) {
                    const int row_new{row + Config::dirs[i][0]};
                    const int col_new{col + Config::dirs[i][1]};
                    if (!opp_visited[row_new][col_new]) {
                        node_queue.emplace(std::vector<int>{row_new,col_new});
                        opp_visited[row_new][col_new] = true;
                    }
                }
            }
        }
        if (area>opp_area) { win_state = -1; }
        else if (area<opp_area) { win_state = 1; }
        is_over = true;
    }
    return legal_moves.get_buffer();
}

void Game_controller::make_move(int move, std::vector<int>* move_history)
{
    const int row{move/(Config::size*4)};
    const int col{(move/4)%Config::size};
    const int dir{move%4};
    current_player ^= 1;
    board[{pos[current_player][0],pos[current_player][1],current_player}] = 0;
    pos[current_player][0] = row;
    pos[current_player][1] = col;
    board[{row,col,current_player}] = 1;
    board[{row,col,dir+2}] = 1;
    board[{row+Config::dirs[dir][0],col+Config::dirs[dir][1],((dir+2)%4)+2}] = 1;
    if (move_history) {
        move_history->insert(move_history->end(),{row,col,dir});
    }
    num_moves += 1;
}

void Game_controller::print_board() const
{
    std::vector<std::string> string_board;
    for (int k{0}; k < 2 * Config::size; ++k) {
        int i{k / 2};
        if (!(k % 2)) {
            string_board.emplace_back("  ");
            for (int j{0}; j < Config::size; ++j) {
                if (board.read({i, j, 2}) == 1) {
                    string_board.emplace_back("   ");
                } else {
                    string_board.emplace_back("|  ");
                }
            }
        } else {
            for (int j{0}; j < Config::size; ++j) {
                if (board.read({i, j, 5}) == 1) {
                    string_board.emplace_back("  ");
                } else {
                    string_board.emplace_back("--");
                }
                if (board.read({i, j, 0}) == 1) {
                    string_board.emplace_back("1");
                } else if (board.read({i, j, 1}) == 1) {
                    string_board.emplace_back("2");
                } else {
                    string_board.emplace_back("*");
                }
            }
            if (board.read({i, Config::size - 1, 3}) == 1) {
                string_board.emplace_back("  ");
            } else {
                string_board.emplace_back("--");
            }
        }
        string_board.emplace_back("\n");
    }
    string_board.emplace_back("  ");
    for (int j{0}; j < Config::size; ++j) {
        if (board.read({Config::size - 1, j, 4}) == 1) {
            string_board.emplace_back("   ");
        } else {
            string_board.emplace_back("|  ");
        }
    }
    std::ostringstream imploded;
    std::copy(string_board.begin(), string_board.end(), std::ostream_iterator<std::string>(imploded, ""));
    std::cout << imploded.str() << std::endl;
}

void print_board(Ndarray<float>& state)
{
    std::vector <std::string> string_board;
    for (int k{0}; k < 2 * Config::size; ++k) {
        int i{k / 2};
        if (!(k % 2)) {
            string_board.emplace_back("  ");
            for (int j{0}; j < Config::size; ++j) {
                if (state.read({i, j, 2}) == 1) {
                    string_board.emplace_back("   ");
                } else {
                    string_board.emplace_back("|  ");
                }
            }
        } else {
            for (int j{0}; j < Config::size; ++j) {
                if (state.read({i, j, 5}) == 1) {
                    string_board.emplace_back("  ");
                } else {
                    string_board.emplace_back("--");
                }
                if (state.read({i, j, 0}) == 1) {
                    string_board.emplace_back("1");
                } else if (state.read({i, j, 1}) == 1) {
                    string_board.emplace_back("2");
                } else {
                    string_board.emplace_back("*");
                }
            }
            if (state.read({i, Config::size - 1, 3}) == 1) {
                string_board.emplace_back("  ");
            } else {
                string_board.emplace_back("--");
            }
        }
        string_board.emplace_back("\n");
    }
    string_board.emplace_back("  ");
    for (int j{0}; j < Config::size; ++j) {
        if (state.read({Config::size - 1, j, 4}) == 1) {
            string_board.emplace_back("   ");
        } else {
            string_board.emplace_back("|  ");
        }
    }
    std::ostringstream imploded;
    std::copy(string_board.begin(), string_board.end(), std::ostream_iterator<std::string>(imploded, ""));
    std::cout << imploded.str() << std::endl;
}