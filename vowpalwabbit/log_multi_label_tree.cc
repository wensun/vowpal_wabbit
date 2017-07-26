#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <sstream>

#include "reductions.h"
#include "rand48.h"

using namespace std;
using namespace LEARNER;

namespace log_multi_label_tree_nc{

struct top_F_node_pred{
    uint32_t label;
    float avg_rew;
    top_F_node_pred(uint32_t a): label(a), avg_rew(0.001){}
}


struct node{
    uint32_t parent;
    bool internal;
    uint32_t depth;
    uint32_t F;  
    uint32_t base_router;
    uint32_t left;
    uint32_t right;
    double n; //number of examples routed to this node so far.
    double num_all_zeros; //number of examples who has no positive labels in the top_F.

    v_array<double> label_cummulative_rew;
    v_array<top_F_node_pred> top_F_label_info;

    node(){
        F = f;
        parent = 0;
        internal = false;
        depth = 0;
        base_router = 0;
        left = 0;
        right = 0;
        n = 0.001;
        num_all_zeros = 0.001;
        label_cummulative_rew (v_init<double>());
        top_F_label_info (v_init<node_pred>());
    }
};

struct log_multi_label_tree {
    vw* all;
    uint64_t k;    
    v_array<node> nodes;
    size_t F;
    size_t max_routers; 
    size_t max_depth;
};


void init_tree(log_multi_label_tree& b, uint32_t root, uint32_t depth, uint32_t& routers_used){
    if (depth <= b.max_depth){
        uint32_t left_child;
        uint32_t right_child;
        left_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());
        right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());
        b.nodes[root].base_router = routers_used++;

        b.nodes[root].internal = true;
        b.nodes[root].left = left_child;
        b.nodes[left_child].parent = root;
        b.nodes[left_child].depth = depth;
        b.nodes[root].right = right_child;
        b.nodes[right_child].parent = root;
        b.nodes[right_child].depth = depth;

        init_tree(b, left_child, depth+1, routers_used);
        init_tree(b, right_child, depth+1, routers_used);
    }
}


void init_tree (log_multi_label_tree& b){
    uint32_t routers_used = 0;
    b.nodes.push_back(node());
    init_tree (b, 0, 1, routers_used);
    b.max_routers = routers_used;
}

//computes the probability that there exist at least one label with rew = 1.
inline float statistics_of_top_F(double num_all_zeros, double total_num){
    float prob_all_zeros = num_all_zeros / total_num;
    return log(1. - prob_all_zeros + 0.00000001)/log(2);
}

inline size_t find_min_in_top_F(const node& cnode){
    size_t min_loc = 0;
    float min_v = FLT_MAX;
    for (size_t i = 0; i < cnode.top_F_label_info.size(); i++){
        if (cnode.top_F_label_info[i].avg_rew < min_v){
            min_v = cnode.top_F_label_info[i].avg_rew;
            min_loc = i;
        }
    }
    return min_loc;
}

inline bool in_top_F(const node& cnode, uint32_t label, size_t& pos){
    bool in = false;
    pos = 0;
    for (size_t i = 0; i < cnode.top_F_label_info.size(); i++){
        if (cnode.top_F_label_info[i].label == label){
            in = true;
            pos = i;
            return in;
        }
    }
    return in;
}

void update_node(node& cnode, example& ec){
    v_array<uint32_t>& labels = ec.l.multilabels;
    //update node
    cnode.n ++; 
    for (uint32_t lab : labels){
        cnode.label_histogram[lab]++;
    }

    //update top_F information:
    for (size_t i = 0; i < cnode.top_F_label_info.size(); i++){
        uint32_t lab = cnode.top_F_label_info[i].label;
        cnode.top_F_label_info[i].avg_rew = cnode.label_histogram[lab]*1./cnode.n;
    }
    for (uint32_t label : labels){
        size_t label_pos_in_top_F = 0;
        bool in = in_top_F(cnode, label, label_pos_in_top_F);
        if (in == true)
            cnode.top_F_label_info[label_pos_in_top_F].avg_rew = cnode.label_histogram[label]*1./cnode.n;
        else{
            size_t min_loc = find_min_in_top_F(cnode);
            float avg_rew_min_loc = cnode.top_F_label_info[min_loc].avg_rew;
            if (cnode.label_histogram[label]*1./cnode.n > avg_rew_min_loc){
                cnode.top_F_label_info[min_loc].label = label;
                cnode.top_F_label_info[min_loc].avg_rew = cnode.label_histogram[label]*1./cnode.n;
            }
        }
    }
    bool overlap_with_top_F = false;
    for (uint32_t label: labels){
        size_t tmp_loc = 0;
        bool in = in_top_F(cnode, label, tmp_loc);
        if (in == true){
            overlap_with_top_F = true;
            break;
        }
    }
    if (overlap_with_top_F == false)
        cnode.num_all_zeros ++;

}

//to do: implement fake update steps
double fake_update_node(node& cnode, example& ec){ 
    v_array<uint32_t>& labels = ec.l.multilabels;
    return 0. 
}


inline void save_restore_multilabel_info(example& ec, v_array<uint32_t>& tmp_l_multilabels,
            v_array<uint32_t>& tmp_pred_multilabels, bool save){
    if (save == true){
        v_array<uint32_t> tmp_l_multilabels = v_init<uint32_t>;
        v_array<uint32_t> tmp_pred_multilabels=v_init<uint32_t>;
        for (int i = 0; i < ec.l.multilabels.size(); i++)
            tmp_l_multilabels.push_back(ec.l.multilabels[i]);
        for (int i = 0; i < ec.pred.multilabels.size(); i++)
            tmp_pred_multilabels.push_back(ec.pred.multilabels[i]);
    } 
    else{
        ec.l.multilabels.delete_v();
        ec.l.multilabels = v_init<uint32_t>();
        ec.pred.multilabels.delete_v();
        ec.pred.multilabels = v_init<uint32_t>();
        for (int i = 0; i < tmp_l_multilabels.size(); i++)
            ec.l.multilabels.push_back(tmp_l_multilabels[i]);
        for (int i = 0; i < tmp_pred_multilabels.size(); i++)
            ec.pred.multilabels.push_back(tmp_pred_multilabels[i]);
        
        tmp_l_multilabels.delete_v();
        tmp_pred_multilabels.delete_v();
    }           
}



float train_internal_node(log_multi_label_tree&b, base_learner& base, uint32_t cn, example& ec){
    if (b.nodes[cn].internal == false){
        cout<<"Error: try to train a leaf node..."<<endl;
        exit(0);
    }
    v_array<uint32_t> tmp_l_multilabels = v_init<uint32_t>;
    v_array<uint32_t> tmp_pred_multilabels=v_init<uint32_t>;
    save_restore_multilabel_info(ec, tmp_l_multilabels, tmp_pred_multilabels, true);

    MULTICLASS::label_t mc = ec.l.multi;
    uint32_t save_pred = ec.pred.multiclass;

    uint32_t left_child = b.nodes[cn].left_child;
    double nl = b.nodes[left_child].n;
    uint32_t right_child = b.nodes[cn].right_child;
    double nr = b.nodes[right_child].n;

    double all_zeros_left = b.nodes[left_child].num_all_zeros;
    double all_zeros_right = b.nodes[right_child].num_all_zeros;

    double all_zeros_left_p = fake_update_node(b.nodes[left_child], ec);
    double all_zeros_right_p= fake_update_node(b.nodes[right_child],ec);

    double benefit_left = ((nl+1.)/(nl+nr+1.)*statistics_of_top_F(all_zeros_left_p,nl+1.)
                        + nr/(nl+nr+1.)*statistics_of_top_F(all_zeros_right, nr));
    double benefit_right= ((nr+1.)/(nl+nr+1.)*statistics_of_top_F(all_zeros_right_p, nr+1.)
                        + nl/(nr+nl+1.)*statistics_of_top_F(all_zeros_left, nl));
    
    float route_label = benefit_left < benefit_right ? 1.f : -1.f;
    float weight = fabs((float)(benefit_left - benefit_right));
    ec.l.simple = {route_label, weight, 0.};
    base.learn(ec, b.nodes[cn].base_router);

    base.predict(ec, b.nodes[cn].base_router);
    float save_scalar = ec.pred.scalar;

    //restore:
    save_restore_multilabel_info(ec, tmp_l_multilabels, tmp_pred_multilabels, false);
    return save_scalar;
}




}