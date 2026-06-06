#include "Tree_search.h"
#include "Config.h"
#include "Game_controller.h"
#include "Ndarray.h"
#include "Random.h"
#include <atomic>
#include <limits>
#include <random>
#include <cmath>
#include <numeric>
#include <thread>
#include <functional>

void Node::create_children(const Ndarray<float>& priors, int thread, float value, std::vector<bool>&& legals)
{
    legal_moves = std::move(legals);
    float max{priors.read({thread,0})};
    for (int i{0}; i<Config::width; ++i) {
        float temp = priors.read({thread,i});
        child_priors[i] = temp;
        max = (temp>max) ? temp : max;
    }
    for (int i{0}; i<Config::width; ++i) {
        child_priors[i] = (legal_moves[i]) ? std::exp(child_priors[i]-max) : 0.0f;
        children[i] = (legal_moves[i]) ? std::make_unique<Node>(i,this) : nullptr;
        child_total_actions[i] = -value;
    }
    if (const float sum=std::accumulate(child_priors.begin(),child_priors.end(),0.0f); sum!=0.0f) {
        for (float& prior:child_priors) {
            prior = prior/sum;
        }
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
    const float sum{std::accumulate(child_visits.begin(),child_visits.end(),0.0f)};
    Array<float> pi;
    for (int i{0}; i<Config::width; ++i) {
        pi[i] = child_visits[i]/sum;
    }
    return pi;
}

inline void Node::add_noise()
{
    static std::gamma_distribution<float> distribution{Config::alpha,1};
    Array<float> sample;
    for (int i{0}; i<Config::width; ++i) {
        sample[i] = (legal_moves[i]) ? distribution(Random::generator) : 0.0f;
    }
    const float sum{std::accumulate(sample.begin(),sample.end(),0.0f)};
    for (int i{0}; i<Config::width; ++i) {
        child_priors[i] = child_priors[i]*(1-Config::epsilon) + (sample[i]/sum)*Config::epsilon;
    }
}

Node::Array<float> Node::get_mean_action() const
{
    Array<float> mean_action;
    for (int i{0}; i<Config::width; ++i) {
        mean_action[i] = (!child_visits[i]) ? child_total_actions[i] : child_total_actions[i]/child_visits[i];
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
        scores[i] = mean_action[i]+c*child_priors[i]*std::sqrt(sum)/(1+child_visits[i]);
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
    delete model;
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
    const int move{root->choose_move(temperature && (game->num_moves<Config::num_high_temp_moves))};
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
        const std::vector<cppflow::tensor> outputs = (*model)({{"serving_default_board:0",inputs.get_tensor()}},
                                                   {"StatefulPartitionedCall:0","StatefulPartitionedCall:1"});
        output_values = outputs[1].get_data<float>();
        output_policies = Ndarray<float>{outputs[0].get_data<float>(),{Config::num_threads,Config::width}};
        output_ready.store(true,std::memory_order_release);
        output_ready.notify_all();
    }
}

void Tree::search(int thread_id)
{
    while (true) {
        output_ready.wait(true,std::memory_order_acquire);
        if (exit.load(std::memory_order_relaxed)) { break; }
        int x{};
        int y{};
        Game_controller game_copy{*game};
        Node* current_node{root.get()};
        while (current_node->expanded) {
            x = current_node->choose_child();
            current_node = current_node->children[x].get();
            current_node->update(-1); // Adding virtual loss, before was just incrementing visits
            current_node->increment_visit(1);
            y = game_copy.make_move(x);
        }
        const bool first = [&] {
            if (!current_node->flag.exchange(true,std::memory_order_relaxed) && !current_node->terminal) {
                int result = game_copy.get_result(x,y);
                if (result == 2) {
                    game_copy.get_state(inputs,thread_id);
                    return true;
                }
                current_node->terminal = true;
                current_node->win_state = result;
            }
            return false;
        }();
        if (ready_count.fetch_add(1,std::memory_order_release) == Config::num_threads-1) {
            ready_count.store(0,std::memory_order_relaxed);
            input_ready.store(true,std::memory_order_release);
            input_ready.notify_one();
        }
        output_ready.wait(false,std::memory_order_acquire);

        if (first || current_node->terminal) {
            float value = [&] {
                if (first) {
                    float value{output_values[thread_id]};
                    current_node->create_children(output_policies,thread_id,value,game_copy.get_legal_moves());
                    return value;
                }
                return static_cast<float>(current_node->win_state);
            }();
            while (current_node->parent) {
                current_node->update(1+value);
                current_node = current_node->parent;
                value *= -1;
            }
        }
        else {
            while (current_node->parent) {
                current_node->update(1);
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