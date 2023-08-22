#include "Tree_search.h"
#include "Config.h"
#include "Game_controller.h"
#include "Ndarray.h"
#include "Random.h"
#include <atomic>
#include <limits>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <functional>

void Node::create_children(Array<float>& priors, const Array<bool>& legals)
{
    child_priors = priors;
    legal_moves = legals;
    const float max{*std::max_element(child_priors.begin(),child_priors.end())};
    for (int i{0}; i<Config::width; ++i) {
        priors[i] = (legal_moves[i]) ? std::exp(priors[i]-max) : 0.0f;
    }

    if (const float sum=std::accumulate(priors.begin(),priors.end(),0.0f); sum!=0.0f) {
        for (float& prior:priors) {
            prior = prior/sum;
        }
    }

    for (int i{0}; i<Config::width; ++i) {
        children[i] = (legal_moves[i]) ? std::make_unique<Node>(i,this) : nullptr;
    }
    expanded = true;
}

inline void Node::increment_visit(int n)
{
    std::atomic_ref<int>{parent->child_visits[move]}.fetch_add(n,std::memory_order_relaxed);
}

inline void Node::update(float value)
{
    std::atomic_ref<float>{parent->child_total_actions[move]}.fetch_add(value,std::memory_order_relaxed);
}

int Node::choose_child() const
{
    int chosen_child;
    float highest_score{std::numeric_limits<float>::lowest()};
    const auto scores{get_appraisal()};
    for (int i{0}; const bool& is_legal:legal_moves) {
        if (is_legal && scores[i]>=highest_score) {
            chosen_child = i;
            highest_score = scores[i];
        }
        ++i;
    }
    return chosen_child;
}

inline int Node::choose_move(int temperature) const
{
    return (temperature) ? high_temp_move() : low_temp_move();
}

Node::Array<float> Node::get_policy() const
{
    const int max{*std::max_element(child_visits.begin(),child_visits.end())};
    Array<float> pi;
    for (int i{0}; i<Config::width; ++i) {
        pi[i] = std::exp(static_cast<float>(child_visits[i]-max));
    }

    for (const float sum{std::accumulate(pi.begin(),pi.end(),0.0f)}; float& value:pi) {
        value = value/sum;
    }
    return pi;
}

inline void Node::add_noise()
{
    static std::gamma_distribution<float> distribution{Config::alpha,1};
    for (auto& prior:child_priors) {
        prior = prior*(1-Config::epsilon) + distribution(Random::generator)*Config::epsilon;
    }
}

Node::Array<float> Node::get_mean_action() const
{
    Array<float> mean_action;
    for (int i{0}; i<Config::width; ++i) {
        mean_action[i] = (!child_visits[i]) ? 0.0f : child_total_actions[i]/child_visits[i];
    }
    return mean_action;
}

Node::Array<float> Node::get_appraisal() const
{
    const float sum{std::accumulate(child_visits.begin(),child_visits.end(),0.0f)};
    const float c{std::log((sum+Config::c_base+1.0f)/Config::c_base)+Config::c_init};

    const auto mean_action{get_mean_action()};
    Array<float> scores;
    for (int i{0}; i<Config::width; ++i) {
        scores[i] = mean_action[i]+c*child_priors[i]*std::sqrt(sum/(1+child_visits[i]));
    }
    return scores;
}

inline int Node::high_temp_move() const
{
    std::discrete_distribution<int> distribution{child_visits.begin(),child_visits.end()};
    return distribution(Random::generator);
}

int Node::low_temp_move() const
{
    int chosen_move;
    int highest_visits{0};
    float highest_score{std::numeric_limits<float>::lowest()};
    const auto scores{get_appraisal()};
    for (int i{0}; const int& visits:child_visits) {
        if (visits>highest_visits
            || (visits==highest_visits && scores[i]>highest_score)) {
            chosen_move = i;
            highest_visits = visits;
            highest_score = scores[i];
        }
        ++i;
    }
    return chosen_move;
}

Tree::~Tree()
{
    exit.store(true,std::memory_order_release);
    output_ready.store(false,std::memory_order_release);
    output_ready.notify_all();
    evaluator.join();
    for (auto& thread:searchers) {
        thread.join();
    }
}

int Tree::choose_move(std::vector<float>* policy_history, int temperature)
{
    if (noise && root->expanded) {
        root->add_noise();
    }
    for (int _{0}; _<Config::num_simulations/Config::num_threads; ++_) {
        output_ready.store(false,std::memory_order_release);
        output_ready.notify_all();
        done.wait(false,std::memory_order_acquire);
        done.store(false,std::memory_order_relaxed);
    }
    const int move{root->choose_move((game->num_moves <= Config::num_high_temp_moves) && temperature)};
    if (policy_history) {
        const auto policy{root->get_policy()};
        policy_history->insert(policy_history->end(),policy.begin(),policy.end());
    }
    root = std::move(root->children[move]);
    root->parent = nullptr;
    return move;
}

void Tree::opponent_move(int move)
{
    if (root->expanded) {
        root = std::move(root->children[move]);
        root->parent = nullptr;
    }
}

void Tree::evaluate()
{
    while (true) {
        output_ready.wait(true,std::memory_order_acquire);
        if (exit.load(std::memory_order_relaxed)) { break; }
        input_ready.wait(false,std::memory_order_acquire);
        input_ready.store(false,std::memory_order_relaxed);
        const std::vector<cppflow::tensor> outputs = model({{"serving_default_board:0",inputs.get_tensor()}},
                                                   {"StatefulPartitionedCall:0","StatefulPartitionedCall:1"});
        output_values = outputs[1].get_data<float>();
        const Ndarray<float> policies{outputs[0].get_data<float>(),{Config::num_threads,Config::width}};
        for (int i{0}; i<Config::num_threads; ++i) {
            for (int j{0}; j<Config::width; ++j) {
                output_policies[i][j] = (to_flip[i]) ? policies[{i,Config::width-j-1}] : policies[{i,j}];
            }
        }
        output_ready.store(true,std::memory_order_release);
        output_ready.notify_all();
    }
}

void Tree::search(int thread_id)
{
    while (true) {
        output_ready.wait(true,std::memory_order_acquire);
        if (exit.load(std::memory_order_relaxed)) { break; }
        Game_controller game_copy{*game};
        Node* current_node{root.get()};
        while (current_node->expanded) {
            const int move{current_node->choose_child()};
            current_node = current_node->children[move].get();
            current_node->increment_visit(1);
            game_copy.make_move(move);
        }

        const bool first{!game_copy.is_over && !current_node->flag.exchange(true,std::memory_order_relaxed)};
        if (first) {
            auto board{game_copy.get_state()};
            const int flip = to_flip[thread_id] = Random::get(0,1);
            for (int i{0}; i<Config::height; ++i) {
                for (int j{0}; j<Config::width; ++j) {
                    for (int k{0}; k<2; ++k) {
                        inputs[{thread_id,i,j,k}] = (flip) ? board[{i,Config::width-j-1,k}] : board[{i,j,k}];
                    }
                }
            }
        }
        if (ready_count.fetch_add(1,std::memory_order_release) == Config::num_threads-1) {
            ready_count.store(0,std::memory_order_relaxed);
            input_ready.store(true,std::memory_order_release);
            input_ready.notify_one();
        }
        output_ready.wait(false,std::memory_order_acquire);

        if (first || game_copy.is_over) {
            float value = [&] {
                if (first) {
                    current_node->create_children(output_policies[thread_id],game_copy.get_legal_moves());
                    return output_values[thread_id];
                }
                return static_cast<float>(game_copy.win_state);
            }();
            while (current_node->parent) {
                current_node->update(value);
                current_node = current_node->parent;
                value *= -1;
            }
        }
        else {
            while (current_node->parent) {
                current_node->increment_visit(-1);
                current_node = current_node->parent;
            }
        }
        if (ready_count.fetch_add(1,std::memory_order_release) == Config::num_threads-1) {
            ready_count.store(0,std::memory_order_relaxed);
            done.store(true,std::memory_order_release);
            done.notify_one();
        }
    }
}