#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <time.h>
#include <sstream>

#include "reductions.h"
#include "rand48.h"
#include "vw.h"


using namespace std;
using namespace LEARNER;

namespace memory_tree_rew_ns
{
    ///////////////////////Helper//////////////////////////////
    //////////////////////////////////////////////////////////
    template<typename T> 
    void pop_at_index(v_array<T>& array, uint32_t index)
    {
        if (index >= array.size()){
            cout<<"ERROR: index is larger than the size"<<endl;
            return;
        }
        if (index == array.size() - 1){
            array.pop();
            return;
        }
        for (size_t i = index+1; i < array.size(); i++){
            array[i-1] = array[i];
        }
        array.pop();
        return;
    }

    void copy_example_data(example* dst, example* src, bool no_feat = false)
    { 
        dst->l = src->l;
        dst->l.multi.label = src->l.multi.label;

        copy_array(dst->tag, src->tag);
        dst->example_counter = src->example_counter;
        dst->ft_offset = src->ft_offset;

        if (no_feat == false){
            copy_array(dst->indices, src->indices);
            for (namespace_index c : src->indices)
                dst->feature_space[c].deep_copy_from(src->feature_space[c]);
            dst->ft_offset = src->ft_offset;

            dst->num_features = src->num_features;
        }
        dst->partial_prediction = src->partial_prediction;
        if (src->passthrough == nullptr) dst->passthrough = nullptr;
        else
        { 
            dst->passthrough = new features;
            dst->passthrough->deep_copy_from(*src->passthrough);
        }
        dst->loss = src->loss;
        dst->weight = src->weight;
        dst->total_sum_feat_sq = src->total_sum_feat_sq;
        dst->confidence = src->confidence;
        dst->test_only = src->test_only;
        dst->end_pass = src->end_pass;
        dst->sorted = src->sorted;
        dst->in_use = src->in_use;
    }

    inline void free_example(example* ec)
    {
        ec->tag.delete_v();
        if (ec->passthrough){
            ec->passthrough->delete_v();
        }
        for (int i = 0; i < 256; i++)
            ec->feature_space[i].delete_v();
        ec->indices.delete_v();
        free(ec);
    }

    ////Implement kronecker_product between two examples:
    //kronecker_prod at feature level:
    void diag_kronecker_prod_fs(features& f1, features& f2, float& total_sum_feat_sq, float norm_sq1, float norm_sq2)
    {
        v_array<feature_index> tmp_f2_indicies = v_init<feature_index>();
        copy_array(tmp_f2_indicies, f2.indicies);
        for (size_t i1 = 0; i1 < f1.indicies.size(); i1++){
            size_t i2 = 0;
            for (i2 = 0; i2 < f2.indicies.size(); i2++){
                if (f1.indicies[i1] == f2.indicies[i2]){
                    //f1.values[i1] = f1.values[i1]*f2.values[i2]/pow(norm_sq1*norm_sq2,0.5) - fabs(f1.values[i1]/pow(norm_sq1,0.5) - f2.values[i2]/pow(norm_sq2,0.5));
                    f1.values[i1] = f1.values[i1]*f2.values[i2] / pow(norm_sq1*norm_sq2, 0.5);
                    total_sum_feat_sq += pow(f1.values[i1],2);
                    tmp_f2_indicies[i2] = 0;
                    break;
                }
            }
            if (i2 == f2.indicies.size()){ //f1's index does not appear in f2, namely value of the index in f2 is zero.
                f1.values[i1] = 0.0;
                //f1.values[i1] = 0.0 - fabs(f1.values[i1]/pow(norm_sq1,0.5));
                total_sum_feat_sq += pow(f1.values[i1],2);
            }
        }
        
        /*for (size_t i2 = 0; i2 < tmp_f2_indicies.size(); i2++){
            if (tmp_f2_indicies[i2] != 0){
                float value = 0.0 - fabs(f2.values[i2]/pow(norm_sq2,0.5));
                f1.push_back(value, f2.indicies[i2]);
                total_sum_feat_sq += pow(value, 2);
            }
        }*/
        tmp_f2_indicies.delete_v();
    }
    //kronecker_prod at example level:
    void diag_kronecker_product(example& ec1, example& ec2, example& ec)
    {
        //ec <= ec1 X ec2
        copy_example_data(&ec, &ec1);
        ec.sorted = false;
        ec.total_sum_feat_sq = 0.0;
        for(namespace_index c : ec.indices){
            for(namespace_index c2 : ec2.indices){
                if (c == c2)
                    diag_kronecker_prod_fs(ec.feature_space[c], ec2.feature_space[c2], ec.total_sum_feat_sq, ec1.total_sum_feat_sq, ec2.total_sum_feat_sq);
            }
        }
    }

    //construct node for tree.
    struct node
    {
        uint32_t parent; //parent index
        int internal;
        //bool internal; //an internal or leaf
        uint32_t depth; //depth.
        uint32_t base_router; //use to index router.
        uint32_t left;  //left child.
        uint32_t right; //right child.

        double nl; //number of examples routed to left.
        double nr; //number of examples routed to right.
        
        double bar_rew_L; //cummulative rewards of examples routed to left.
        double bar_rew_R; //cummulative rewards of examples routed to right.

        v_array<uint32_t> examples_index;

        node (){//construct:
            parent = 0;
            internal = 0; //0:not used, 1:internal, -1:leaf 
            depth = 0;
            base_router = 0;
            left = 0;
            right = 0;
            nl = 0.001; //initilze to 1, as we need to do nl/nr.
            nr = 0.001; 
            bar_rew_L = 0.;
            bar_rew_R = 0.;
            examples_index = v_init<uint32_t>();
        }
    };

    //memory_tree
    struct memory_tree
    {
        vw* all;

        v_array<node> nodes;  //array of nodes.
        v_array<example*> examples; //array of example points
        v_array<float> rewards; //corresponding rewards for examples after insertion. 

        size_t max_leaf_examples; 
        size_t max_nodes;
        size_t max_routers;
        float alpha; //for cpt type of update.
        float lambda;
        size_t routers_used;
        int iter;

        size_t max_depth;
        size_t max_ex_in_leaf;

        bool path_id_feat;

        uint32_t num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;
        bool learn_at_leaf;

        bool test_mode;

        memory_tree(){
            nodes = v_init<node>();
            examples = v_init<example*>();
            rewards = v_init<float>();
            alpha = 0.5;
            lambda = 0.0;
            routers_used = 0;
            iter = 0;
            num_mistakes = 0;
            num_ecs = 0;
            num_test_ecs = 0;
            path_id_feat = false;
            test_mode = false;
            max_depth = 0;
            max_ex_in_leaf = 0;
        }
    };

    float linear_kernel(const flat_example* fec1, const flat_example* fec2)
    { 
        float dotprod = 0;
        features& fs_1 = (features&)fec1->fs;
        features& fs_2 = (features&)fec2->fs;
        if (fs_2.indicies.size() == 0)
            return 0.f;
   
        int numint = 0;
        for (size_t idx1 = 0, idx2 = 0; idx1 < fs_1.size() && idx2 < fs_2.size() ; idx1++)
        { 
            uint64_t ec1pos = fs_1.indicies[idx1];
            uint64_t ec2pos = fs_2.indicies[idx2];
            if(ec1pos < ec2pos) continue;
   
            while(ec1pos > ec2pos && ++idx2 < fs_2.size())
                ec2pos = fs_2.indicies[idx2];
   
            if(ec1pos == ec2pos)
            { 
                numint++;
                dotprod += fs_1.values[idx1] * fs_2.values[idx2];
                ++idx2;
            }
        }
        return dotprod;
    }

    float normalized_linear_prod(memory_tree& b, example* ec1, example* ec2)
    {
        flat_example* fec1 = flatten_sort_example(*b.all, ec1);
        flat_example* fec2 = flatten_sort_example(*b.all, ec2);
        float linear_prod = linear_kernel(fec1, fec2);
        fec1->fs.delete_v(); 
        fec2->fs.delete_v();
        free(fec1);
        free(fec2);
        return linear_prod/pow(fec1->total_sum_feat_sq*fec2->total_sum_feat_sq, 0.5);
    }

    void init_tree(memory_tree& b)
    {
        srand48(time(0));
        //simple initilization: initilize the root only
        b.routers_used = 0;
        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.nodes[0].base_router = (b.routers_used++);

        b.max_routers = b.max_nodes;
        cout<<"tree initiazliation is done...."<<endl
            <<"max nodes "<<b.max_nodes<<endl
            <<"tree size: "<<b.nodes.size()<<endl
            <<"learn at leaf: "<<b.learn_at_leaf<<endl;
    }

    //return the id of the example and the leaf id (stored in cn)
    inline int random_sample_example_pop(memory_tree& b, uint32_t& cn, bool decrease_count = true)
    {   //always start from the root: root is initialized at cn:
        while (b.nodes[cn].internal == 1)
        {
            float pred = 0.;   //deal with some edge cases:
            if (b.nodes[cn].nl < 1) //no examples routed to left ever:
                pred = 1.f; //go right.
            else if (b.nodes[cn].nr < 1) //no examples routed to right ever:
                pred = -1.f; //go left.
            else if ((b.nodes[cn].nl >= 1) && (b.nodes[cn].nr >= 1))
                pred = merand48(b.all->random_state) < (b.nodes[cn].nl*1./(b.nodes[cn].nr+b.nodes[cn].nl)) ? -1.f : 1.f;
            else{
                cout<<cn<<" "<<b.nodes[cn].nl<<" "<<b.nodes[cn].nr<<endl;
                cout<<"Error:  nl = 0, and nr = 0, exit...";
                exit(0);
            }
            if (pred < 0){
                if (decrease_count == true)
                    b.nodes[cn].nl--;
                cn = b.nodes[cn].left; 
            }
            else{
                if (decrease_count == true)
                    b.nodes[cn].nr--;
                cn = b.nodes[cn].right;
            }
        }
        if (b.nodes[cn].examples_index.size() >= 1){
            int loc_at_leaf = int(merand48(b.all->random_state)*b.nodes[cn].examples_index.size());
            if (decrease_count == true)
                pop_at_index(b.nodes[cn].examples_index, loc_at_leaf); 
            return loc_at_leaf;
        }
        else    //leaf that has zero examples. 
            return -1;
    }

    float to_prob (float x)
    { 
        static const float alpha = 2.0f;
        // http://stackoverflow.com/questions/2789481/problem-calling-stdmax
        //return (std::max) (0.f, (std::min) (1.f, 0.5f * (1.0f + alpha * x)));
        return exp(x)/(1.+exp(x));

        //if (x > 0)
        //    return 1.;
        //else
        //    return 0.;
    }

    //randomize the routing procedure.
    //lambda = 1: pure random sample; lambda = 0: random based on prediction only.
    inline int random_routing_to_leaf(memory_tree& b, base_learner& base, 
            uint32_t& cn, example& ec, float& sample_p)
    {
        sample_p = 1.0;
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_pred = ec.pred.multiclass;
        ec.l.simple = {FLT_MAX, 1.0, 0.};
        while (b.nodes[cn].internal == 1){
            base.predict(ec, b.nodes[cn].base_router);
            float prob_right = to_prob(ec.pred.scalar);
            //cout<<prob_right<<endl;
            //exp(ec.pred.scalar)/(1.+exp(ec.pred.scalar));
            prob_right = (1.-b.lambda)*prob_right + b.lambda*(b.nodes[cn].nr/(b.nodes[cn].nr+b.nodes[cn].nl+0.00001));
            if (merand48(b.all->random_state) <= prob_right){
                cn = b.nodes[cn].right;
                sample_p *= prob_right;
            }
            else{
                cn = b.nodes[cn].left;
                sample_p *= (1. - prob_right);
            }
        }
        ec.l.multi = mc;
        ec.pred.multiclass = save_pred;
        if (b.nodes[cn].examples_index.size() >= 1){
            int loc_at_leaf = int(merand48(b.all->random_state)*b.nodes[cn].examples_index.size());
            sample_p *= 1./(b.nodes[cn].examples_index.size());
            return loc_at_leaf;
        }
        else
            return -1;
    }


    //pick up the "closest" example in the leaf using the score function.
    int64_t pick_nearest(memory_tree& b, base_learner& base, const uint32_t cn, example& ec, bool uniform_sample = false)
    {
        if (b.nodes[cn].examples_index.size() > 0 && uniform_sample == false)
        {
            float max_score = -FLT_MAX;
            int64_t max_pos = -1;
            for(size_t i = 0; i < b.nodes[cn].examples_index.size(); i++)
            {
                float score = 0.f;
                uint32_t loc = b.nodes[cn].examples_index[i];

                if (b.learn_at_leaf == true){
                    //cout<<"learn at leaf"<<endl;
                    float tmp_s = normalized_linear_prod(b, &ec, b.examples[loc]);
                    example* kprod_ec = &calloc_or_throw<example>();
                    diag_kronecker_product(ec, *b.examples[loc], *kprod_ec);
                    kprod_ec->l.simple = {-1., 1., tmp_s};
                    base.predict(*kprod_ec, b.max_routers);
                    score = kprod_ec->partial_prediction;
                    free_example(kprod_ec);
                }
                else
                    score = normalized_linear_prod(b, &ec, b.examples[loc]);
                
                if (score > max_score){
                    max_score = score;
                    max_pos = (int64_t)loc;
                }
            }
            return max_pos;
        }
        else if(b.nodes[cn].examples_index.size () > 0 && uniform_sample == true) //uniform sample from the leaf.
        {
            int loc_at_leaf = int(merand48(b.all->random_state)*b.nodes[cn].examples_index.size());
            int64_t pos = (int64_t)b.nodes[cn].examples_index[loc_at_leaf];
            return pos;
        }
        else
            return -1;
    }


    inline uint32_t routing(memory_tree& b, const uint32_t& root, base_learner& base, example& ec)
    {   
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        uint32_t cn = root;
        ec.l.simple = {-1.f, 1.f, 0.};
        while(b.nodes[cn].internal == 1){
            base.predict(ec, b.nodes[cn].base_router);
            uint32_t newcn = ec.pred.scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right;
            cn = newcn;
        }
        ec.l.multi = mc;
        ec.pred.multiclass = save_multi_pred;
        return cn;
    }

    inline float get_reward(const example& ec, const example& retrieved_ec)
    {
        //for multi-class example:
        if (ec.l.multi.label == retrieved_ec.l.multi.label)
            return 1.f;
        else
            return 0.f;
    }

    float routing_grab_reward(memory_tree& b, base_learner& base, example& ec, const uint32_t cn, uint32_t current_d, const uint32_t max_d)
    {
        if (b.nodes[cn].internal == -1){
            int64_t closest = pick_nearest(b, base, cn, ec);
            float reward = closest == -1 ? 0 : get_reward(ec, *b.examples[closest]);
            return reward;
        }
        else if (current_d == max_d){
            float reward = 0.;
            uint32_t routed_leaf = routing(b, cn, base, ec);
            int64_t closest_ec = pick_nearest(b, base, routed_leaf, ec);
            reward = closest_ec == -1? 0.:get_reward(ec, *b.examples[closest_ec]);
            return reward;
        }
        else if (current_d < max_d){
            uint32_t left_c = b.nodes[cn].left;
            uint32_t right_c = b.nodes[cn].right;
            float reward_l = routing_grab_reward(b, base, ec, left_c, current_d+1, max_d);
            float reward_r = routing_grab_reward(b, base, ec, right_c, current_d+1, max_d);
            float reward = reward_l > reward_r ? reward_l : reward_r;
            return reward;
        }
        return 0.;
    }

    //memory_tree& b, base_learner& base, 
    //uint32_t& cn, example& ec, float& sample_p, 
    float train_node(memory_tree& b, base_learner& base, example& ec, const uint32_t cn)
    {
        float sample_repeat = log(b.max_nodes/2.*b.max_leaf_examples)/log(2.)*1.;
        float beta = 0.99;
        if (b.nodes[cn].internal == -1){ //leaf: do nothing. 
            cout<<"Error: Train_node is called at a leaf.."<<endl;
            return 0.f;
        }
        float rew_left = 0.;
        float rew_right = 0.;
        for (int repeat = 0; repeat <= sample_repeat; repeat++){
            uint32_t cn_left = b.nodes[cn].left;
            uint32_t cn_right = b.nodes[cn].right;
            float left_sample_p = 1.0; //int loc_at_left_leaf = random_sample_example_pop(b, cn_left, false);
            int loc_at_left_leaf = random_routing_to_leaf(b, base, cn_left, ec, left_sample_p);
            float right_sample_p = 1.0; //int loc_at_right_leaf = random_sample_example_pop(b, cn_right, false);
            int loc_at_right_leaf= random_routing_to_leaf(b, base, cn_right,ec, right_sample_p);
            rew_left += (loc_at_left_leaf == -1 ? 0.:get_reward(ec, *b.examples[b.nodes[cn_left].examples_index[loc_at_left_leaf]]))/left_sample_p/b.nodes[cn].nl;
            rew_right+= (loc_at_right_leaf== -1 ? 0.:get_reward(ec, *b.examples[b.nodes[cn_right].examples_index[loc_at_right_leaf]]))/right_sample_p/b.nodes[cn].nr;
        }
        rew_left /= (sample_repeat*1.0);
        rew_right /= (sample_repeat*1.0);

        //r conditioned right:
        float r_R_p = (b.nodes[cn].nr+1.)/(b.nodes[cn].nr+1.+b.nodes[cn].nl)*(beta*b.nodes[cn].bar_rew_R+(1.-beta)*rew_right)/(1.-pow(beta,b.nodes[cn].nr+1));
        float r_R = (b.nodes[cn].nr)/(b.nodes[cn].nl+b.nodes[cn].nr+1.0)*b.nodes[cn].bar_rew_R/(1.-pow(beta,b.nodes[cn].nr));
        float r_L_p = (b.nodes[cn].nl+1.0)/(b.nodes[cn].nr+b.nodes[cn].nl+1.)*(beta*b.nodes[cn].bar_rew_L+(1.-beta)*rew_left)/(1.-pow(beta,b.nodes[cn].nl+1));
        float r_L = (b.nodes[cn].nl)/(b.nodes[cn].nl+b.nodes[cn].nr+1.0)*b.nodes[cn].bar_rew_L/(1.-pow(beta,b.nodes[cn].nl));
        float delta_r_l = (r_R_p + r_L) - (r_L_p + r_R);
        float balanced = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(delta_r_l); //regularization for balance.
        float route_label = (balanced <= 0 ? -1.f : 1.f);

        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        ec.l.simple = {route_label, fabs(balanced), 0.f};
        base.learn(ec, b.nodes[cn].base_router);
        base.predict(ec, b.nodes[cn].base_router);
        float save_pred_scalar = ec.pred.scalar;
        
        ec.l.multi = mc;
        ec.pred.multiclass = save_multi_pred;
        //update the bar_rew in nodes based on partial prediction: update exponential moving average:
        if (save_pred_scalar <= 0)
            b.nodes[cn].bar_rew_L = beta*b.nodes[cn].bar_rew_L + (1.-beta)*rew_left;
        else
            b.nodes[cn].bar_rew_R = beta*b.nodes[cn].bar_rew_R + (1.-beta)*rew_right;
        return save_pred_scalar;
    }

    void split_leaf(memory_tree& b, base_learner& base, const uint32_t cn)
    {
        //create two children
        b.nodes[cn].internal = 1; //swith to internal node.
        b.nodes[cn].nl = 0.001;
        b.nodes[cn].nr = 0.001;
        
        uint32_t left_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());
        b.nodes[left_child].internal = -1;  //left leaf
        b.nodes[left_child].base_router = (b.routers_used++);
        uint32_t right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());  
        b.nodes[right_child].internal = -1;  //right leaf
        b.nodes[right_child].base_router = (b.routers_used++); 

        b.nodes[cn].left = left_child;
        b.nodes[cn].right = right_child;
        b.nodes[left_child].parent = cn;
        b.nodes[right_child].parent = cn;
        b.nodes[left_child].depth = b.nodes[cn].depth + 1;
        b.nodes[right_child].depth = b.nodes[cn].depth + 1;

        if (b.nodes[cn].depth+1 > b.max_depth){
            b.max_depth = b.nodes[cn].depth + 1;
            cout<<"max depth increase to "<<b.max_depth<<endl;
        }

        for (size_t ec_id = 0; ec_id < b.nodes[cn].examples_index.size(); ec_id++)
        {
            uint32_t ec_pos = b.nodes[cn].examples_index[ec_id];
            float pred_scalar = train_node(b, base, *b.examples[ec_pos], cn); //train and pred, 
            if (pred_scalar <= 0){ //go left
                b.nodes[left_child].examples_index.push_back(ec_pos);
                b.nodes[cn].nl += 1;    
            }
            else{
                b.nodes[right_child].examples_index.push_back(ec_pos);
                b.nodes[cn].nr += 1;
            }
        }
        b.nodes[cn].examples_index.delete_v();
    }

    void learn_similarity_at_leaf(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        for (uint32_t loc : b.nodes[cn].examples_index)
        {
            example* ec_loc = b.examples[loc];
            example* kprod_ec = &calloc_or_throw<example>();
            float score = normalized_linear_prod(b, &ec, ec_loc);
            diag_kronecker_product(ec, *ec_loc, *kprod_ec);
            float rew = get_reward(ec, *ec_loc);
            //if (rew == 0)
            //    rew = -1.;
            kprod_ec->l.simple = {rew, 1., -score}; //regressor;
            base.learn(*kprod_ec, b.max_routers);
            free_example(kprod_ec);
        }
    }

    void insert_example(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, bool fake_insert = false)
    {
        uint32_t cn = 0;
        while(b.nodes[cn].internal == 1)
        {
            float pred_scalar = train_node(b, base, *b.examples[ec_array_index], cn);
            if (pred_scalar < 0){ //go left:
                b.nodes[cn].nl++;
                cn = b.nodes[cn].left;
            }
            else{//go right
                b.nodes[cn].nr++;
                cn = b.nodes[cn].right;
            }
        }

        if ((b.nodes[cn].internal == -1) && (b.learn_at_leaf == true))
            learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]);
        
        if ((b.nodes[cn].internal == -1) && (fake_insert == false)){
            b.nodes[cn].examples_index.push_back(ec_array_index); //insert this example's index at the examples
            if ((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size()+2 <=b.max_nodes)){
                split_leaf(b, base, cn);
            }
        }
    }

    void experience_replay(memory_tree& b, base_learner& base)
    {
        uint32_t cn = 0; //start from root, randomly descent down! 
        int loc_at_leaf = random_sample_example_pop(b, cn, true);
        if (loc_at_leaf >= 0){
            uint32_t ec_id = b.nodes[cn].examples_index[loc_at_leaf]; //ec_id is the postion of the sampled example in b.examples. 
            insert_example(b, base, ec_id); 
        }
    }

    void predict(memory_tree& b, base_learner& base, example& ec)
    {
        uint32_t cn = routing(b, 0, base, ec);
        int64_t closest_ec = pick_nearest(b,base,cn,ec);
        float rew = (closest_ec == -1 ? 0.f : get_reward(ec, *b.examples[closest_ec]));

        if (closest_ec != -1)
            ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
        
        ec.loss = (1. - rew); //convert reward to loss.
        b.num_mistakes += ec.loss;
    }

    void learn(memory_tree& b, base_learner& base, example& ec)
    {
        b.iter++;
        if (b.test_mode == false){
            predict(b, base, ec); 
            if (b.iter%5000 == 0)
                cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;

            example* new_ec = &calloc_or_throw<example>();
            copy_example_data(new_ec, &ec);
            //remove_repeat_features_in_ec(*new_ec); ////sort unique.
            b.examples.push_back(new_ec);   
            b.num_ecs++; 

            insert_example(b, base, b.examples.size()-1);

            for (int i = 0; i < 1; i++)
                experience_replay(b, base);
        }
        else if (b.test_mode == true){
            if (b.iter % 5000 == 0)
                cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
            predict(b, base, ec);
        }

    }

    void finish(memory_tree& b)
    {
        for (size_t i = 0; i < b.nodes.size(); ++i)
            b.nodes[i].examples_index.delete_v();
        b.nodes.delete_v();
        for (size_t i = 0; i < b.examples.size(); i++)
            free_example(b.examples[i]);
        b.examples.delete_v();
        cout<<b.max_nodes<<endl;
    }

    ///////////////////Save & Load//////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    #define writeit(what,str)                               \
    do                                                    \
        {                                                   \
        msg << str << " = " << what << " ";               \
        bin_text_read_write_fixed (model_file,            \
                                    (char*) &what,         \
                                    sizeof (what),         \
                                    "",                    \
                                    read,                  \
                                    msg,                   \
                                    text);                 \
        }                                                   \
    while (0);

    #define writeitvar(what,str,mywhat)                     \
    auto mywhat = (what);                                 \
    do                                                    \
        {                                                   \
        msg << str << " = " << mywhat << " ";             \
        bin_text_read_write_fixed (model_file,            \
                                    (char*) &mywhat,       \
                                    sizeof (mywhat),       \
                                    "",                    \
                                    read,                  \
                                    msg,                   \
                                    text);                 \
        }                                                   \
    while (0);

    void save_load_example(example* ec, io_buf& model_file, bool& read, bool& text, stringstream& msg)
    {   //deal with tag
        //deal with labels:
        writeit(ec->l.multi.label, "multiclass_label");
        writeit(ec->l.multi.weight, "multiclass_weight");
        writeit(ec->num_features, "num_features");
        writeit(ec->total_sum_feat_sq, "total_sum_features");
        writeit(ec->weight, "example_weight");
        writeit(ec->loss, "loss");
        writeit(ec->ft_offset, "ft_offset");

        writeitvar(ec->tag.size(), "tags", tag_number);
        if (read){
            ec->tag.erase();
            for (uint32_t i = 0; i < tag_number; i++)
                ec->tag.push_back('a');
        }
        for (uint32_t i = 0; i < tag_number; i++)
            writeit(ec->tag[i], "tag");
        
        //deal with tag:
        writeitvar(ec->indices.size(), "namespaces", namespace_size);
        if (read){
            ec->indices.delete_v();
            for (uint32_t i = 0; i< namespace_size; i++){
                ec->indices.push_back('\0');
            }
        }
        for(uint32_t i = 0; i < namespace_size; i++)
            writeit(ec->indices[i], "namespace_index");

        //deal with features
        for (namespace_index nc: ec->indices){
            features* fs = &ec->feature_space[nc];
            writeitvar(fs->size(), "features_", feat_size);
            if (read){
                fs->erase();
                fs->values = v_init<feature_value>();
                fs->indicies = v_init<feature_index>();
                for (uint32_t f_i = 0; f_i < feat_size; f_i++){
                    fs->push_back(0, 0);
                }
            }
            for (uint32_t f_i = 0; f_i < feat_size; f_i++)
                writeit(fs->values[f_i], "value");
            for (uint32_t f_i = 0; f_i < feat_size; f_i++)
                writeit(fs->indicies[f_i], "index");
        }
    }

    void save_load_node(node& cn, io_buf& model_file, bool& read, bool& text, stringstream& msg)
    {
        writeit(cn.parent, "parent");
        writeit(cn.internal, "internal");
        writeit(cn.depth, "depth");
        writeit(cn.base_router, "base_router");
        writeit(cn.left, "left");
        writeit(cn.right, "right");
        writeit(cn.nl, "nl");
        writeit(cn.nr, "nr");
        writeitvar(cn.examples_index.size(), "leaf_n_examples", leaf_n_examples);
        if (read){
            cn.examples_index.erase();
            for (uint32_t k = 0; k < leaf_n_examples; k++)
                    cn.examples_index.push_back(0);
        }
        for (uint32_t k = 0; k < leaf_n_examples; k++)
            writeit(cn.examples_index[k], "example_location");
    }

    void save_load_memory_tree(memory_tree& b, io_buf& model_file, bool read, bool text)
    {
        stringstream msg;
        if (model_file.files.size() > 0){
            if (read)
                b.test_mode = true;

            if (read)
            {
                size_t ss = 0;
                writeit(ss, "stride_shift");
                b.all->weights.stride_shift(ss);
            }
            else
            {
                size_t ss = b.all->weights.stride_shift();
                writeit(ss, "stride_shift");
            }
            
            writeit(b.max_nodes, "max_nodes");
            writeit(b.learn_at_leaf, "learn_at_leaf");
            writeitvar(b.nodes.size(), "nodes", n_nodes); 

            if (read){
                b.nodes.erase();
                for (uint32_t i = 0; i < n_nodes; i++)
                    b.nodes.push_back(node());
            }
            
            //node  
            for(uint32_t i = 0; i < n_nodes; i++){
                save_load_node(b.nodes[i], model_file, read, text, msg);
            }
            //deal with examples:
            writeitvar(b.examples.size(), "examples", n_examples);
            if (read){
                b.examples.erase();
                for (uint32_t i = 0; i < n_examples; i++){
                    example* new_ec = &calloc_or_throw<example>();
                    b.examples.push_back(new_ec);
                }
            }
            for (uint32_t i = 0; i < n_examples; i++)
                save_load_example(b.examples[i], model_file, read, text, msg);
            
            
        }
    }
    //////////////////////////////End of Save & Load///////////////////////////////

} //namespace

base_learner* memory_tree_rew_setup(vw& all)
{
    using namespace memory_tree_rew_ns;
    if (missing_option<uint32_t, true>(all, "memory_tree_rew", "Make a memory tree with at most <n> nodes"))
        return nullptr;
    
    new_options(all, "memory tree (reward guided) options")
      ("leaf_example_multiplier", po::value<uint32_t>()->default_value(1.0), "multiplier on examples per leaf (default = log nodes)")
      ("learn_at_leaf", po::value<bool>()->default_value(true), "whether or not learn at leaf (defualt = True)")
      ("Alpha", po::value<float>()->default_value(0.5), "Alpha")
      ("Lambda", po::value<float>()->default_value(0.1), "Lambda");
     add_options(all);

    po::variables_map& vm = all.vm;
    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.all = &all;
    tree.max_nodes = vm["memory_tree_rew"].as<uint32_t>();
    tree.learn_at_leaf = vm["learn_at_leaf"].as<bool>();

    if (vm.count("leaf_example_multiplier"))
      {
	tree.max_leaf_examples = vm["leaf_example_multiplier"].as<uint32_t>() * (log(tree.max_nodes)/log(2));
	*all.file_options << " --leaf_example_multiplier " << vm["leaf_example_multiplier"].as<uint32_t>();
      }
    if (vm.count("Alpha"))
    {
	    tree.alpha = vm["Alpha"].as<float>();
	    *all.file_options << " --Alpha " << tree.alpha;
    }
    if (vm.count("Lambda"))
    {
        tree.lambda = vm["Lambda"].as<float>();
        *all.file_options << " --Lambda " <<tree.lambda;
    }
    
    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "memory_tree_rew:" << " "
                    <<"max_nodes = "<< tree.max_nodes << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
                    <<"alpha = "<<tree.alpha<<" "
                    <<"Lambda = "<<tree.lambda
                    <<std::endl;
    
    learner<memory_tree>& l = 
        init_multiclass_learner (&tree, 
                setup_base (all),
                learn,
                predict,
                all.p, 
                tree.max_nodes + 1);
    
    l.set_save_load(save_load_memory_tree);
    l.set_finish(finish);

    return make_base (l);
}





///////
/*
float train_node_old(memory_tree& b, base_learner& base, example& ec, const uint32_t cn, bool uniform_sample = false)
{
    if (b.nodes[cn].internal == -1){ //leaf: do nothing. 
        cout<<"Error: Train_node is called at a leaf.."<<endl;
        return 0.f;
    }
    float rew_left = routing_grab_reward(b, base, ec, b.nodes[cn].left, 0, 2);
    float rew_right = routing_grab_reward(b, base, ec, b.nodes[cn].right, 0, 2);
     
    float delta_r_l = rew_right - rew_left; 
    float balanced = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(delta_r_l); //regularization for balance.
    float route_label = (balanced <= 0 ? -1.f : 1.f);

    MULTICLASS::label_t mc = ec.l.multi;
    uint32_t save_multi_pred = ec.pred.multiclass;
    ec.l.simple = {route_label, fabs(balanced), 0.f};
    //ec.l.simple = {route_label, 1.f, 0.f};
    base.learn(ec, b.nodes[cn].base_router);
    base.predict(ec, b.nodes[cn].base_router);
    float save_pred_scalar = ec.pred.scalar;
    
    ec.l.multi = mc;
    ec.pred.multiclass = save_multi_pred;
    return save_pred_scalar;
}*/