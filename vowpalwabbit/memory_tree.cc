#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <time.h>
#include <sstream>
#include <ctime>

#include "reductions.h"
#include "rand48.h"
#include "vw.h"


using namespace std;
using namespace LEARNER;

namespace memory_tree_ns
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



    void remove_repeat_features_in_f(features& f, float& total_sum_feat_sq )
    {
        for (size_t i = 0; i < f.indicies.size(); i++){
            if (f.values[i] != -FLT_MAX){
                uint64_t loc = f.indicies[i];
                for (size_t j = i+1; j < f.indicies.size(); j++){
                    if (loc == f.indicies[j]){
                        total_sum_feat_sq -= (f.values[i]*f.values[i] + f.values[j]*f.values[j]);
                        f.values[i] += f.values[j];
                        total_sum_feat_sq += f.values[i]*f.values[i];
                        f.values[j] = -FLT_MAX;
                    }
                }
            }
        }
        for (size_t i = 0; i < f.indicies.size(); i++){
            if (f.values[i] == -FLT_MAX)
                f.values[i] = 0.0;
        }
    }
    
    void remove_repeat_features_in_ec(example& ec)
    {
        for (auto nc : ec.indices)
            remove_repeat_features_in_f(ec.feature_space[nc], ec.total_sum_feat_sq);
    }

    //f = f1 - f2
    void subtract_features(features& f1, features& f2, features& f)
    {//f1 and f2 are features under the same namespace
        f.deep_copy_from(f1);
        //now scan through f2's indices, check if the index is in f1's indices:
        for (size_t i = 0; i < f2.indicies.size(); i++){
            uint64_t ind_f2 = f2.indicies[i];
            float val_f2 = f2.values[i];
            uint64_t pos = 0;
            for (pos = 0; pos < f.indicies.size(); pos++)
            {
                if (ind_f2 == f.indicies[pos]) //the feature index of f2 is also in f1, then we subtract:
                {
                    f.values[pos] -= val_f2;
                    break;
                }
            }
            if (pos == f.indicies.size()) //this feature index of f2 is not in f1,  then push back (0-val), as we are doing f1-f2.
                f.push_back(0. - val_f2, ind_f2); 
        }
    }


    //ec1 - ec2
    void subtract_two_examples(example& ec1, example& ec2, example* subtracted_ec)
    {
        //copy tag from ec1 and ec2:
        copy_array(subtracted_ec->tag, ec1.tag);
        for (auto tag: ec2.tag)  //char
        {
	  size_t i = 0;
            for(i = 0; i < subtracted_ec->tag.size(); i++){
                if (tag == subtracted_ec->tag[i]) 
                    break;
            }
            if (i == subtracted_ec->tag.size())
                subtracted_ec->tag.push_back(tag);
        }
        //copy indices (different indices, no repeat)
        copy_array(subtracted_ec->indices, ec1.indices);
        for (auto c : ec2.indices){ //char: namespace_index
            size_t i = 0;
            for (i = 0; i < subtracted_ec->indices.size(); i++){
                if (c == subtracted_ec->indices[i])
                    break;
            }
            if (i == subtracted_ec->indices.size())
                subtracted_ec->indices.push_back(c);
        }
        //copy features (f1 - f2) 
        subtracted_ec->ft_offset = ec1.ft_offset; //0
        subtracted_ec->num_features = 0;
        subtracted_ec->total_sum_feat_sq = 0; 
        for (auto c : subtracted_ec->indices) //namespace index:
        {
            //cout<<"at "<<c<<endl;
            size_t pos1 = 0;
            for (pos1 = 0; pos1 < ec1.indices.size(); pos1++){
                if (c == ec1.indices[pos1])
                    break;
            }
            size_t pos2 = 0;
            for (pos2 = 0; pos2 < ec2.indices.size(); pos2++){
                if (c == ec2.indices[pos2])
                    break;
            }

            if ((pos1 < ec1.indices.size()) && (pos2 < ec2.indices.size())) //common name space:
                subtract_features(ec1.feature_space[c], ec2.feature_space[c], subtracted_ec->feature_space[c]);
            else if ((pos1 < ec1.indices.size()) && (pos2 == ec2.indices.size())) //f1 has this a name space that doesn't exist in f2:
                subtracted_ec->feature_space[c].deep_copy_from(ec1.feature_space[c]);
            else if ((pos1 == ec1.indices.size()) && (pos2 < ec2.indices.size())){
                subtracted_ec->feature_space[c].deep_copy_from(ec2.feature_space[c]);
                //negate the values
                for (size_t t = 0; t < subtracted_ec->feature_space[c].values.size(); t++)
                    subtracted_ec->feature_space[c].values[t] *= -1.;
            }
            subtracted_ec->num_features += subtracted_ec->feature_space[c].indicies.size();

            //update total_feature_square
            for (size_t v = 0; v < subtracted_ec->feature_space[c].values.size(); v++){
                subtracted_ec->total_sum_feat_sq += pow(subtracted_ec->feature_space[c].values[v],2);
                //cout<<pow(subtracted_ec->feature_space[c].values[v],2)<<endl;
            }
        }

        //otherstuff:
        subtracted_ec->partial_prediction = 0.0;
        subtracted_ec->passthrough = nullptr;
        subtracted_ec->loss = 0.;
        subtracted_ec->weight = ec1.weight;
        subtracted_ec->confidence = ec1.confidence;
        subtracted_ec->test_only = ec1.test_only;
        subtracted_ec->end_pass = ec1.end_pass;
        subtracted_ec->sorted = false;
        subtracted_ec->in_use = false;
    }


    ////Implement kronecker_product between two examples:
    //kronecker_prod at feature level:

    void diag_kronecker_prod_fs_test(features& f1, features& f2, features& prod_f, float& total_sum_feat_sq, float norm_sq1, float norm_sq2)
    {
        prod_f.delete_v();
        if (f2.indicies.size() == 0)
            return;
    
        for (size_t idx1 = 0, idx2 = 0; idx1 < f1.size() && idx2 < f2.size(); idx1++)
        {
            uint64_t ec1pos = f1.indicies[idx1];
            uint64_t ec2pos = f2.indicies[idx2];
            if (ec1pos < ec2pos) continue;

            while (ec1pos > ec2pos && ++idx2 < f2.size())
                ec2pos = f2.indicies[idx2];

            if (ec1pos == ec2pos){
                prod_f.push_back(f1.values[idx1]*f2.values[idx2]/pow(norm_sq1*norm_sq2,0.5), ec1pos);
                total_sum_feat_sq+=f1.values[idx1]*f2.values[idx2]/pow(norm_sq1*norm_sq2,0.5);
                ++idx2;
            }
        }
    }

    void diag_kronecker_product_test(example& ec1, example& ec2, example& ec)
    {
        copy_example_data(&ec, &ec1);
        ec.total_sum_feat_sq = 0.0;
        for(namespace_index c1 : ec1.indices){
            for(namespace_index c2 : ec2.indices){
                if (c1 == c2)
                    diag_kronecker_prod_fs_test(ec1.feature_space[c1], ec2.feature_space[c2], ec.feature_space[c1], ec.total_sum_feat_sq, ec1.total_sum_feat_sq, ec2.total_sum_feat_sq);
            }
        }
    }


    /*
    void diag_kronecker_prod_fs(features& f1, features& f2, float& total_sum_feat_sq, float norm_sq1, float norm_sq2)
    {
        v_array<feature_index> tmp_f2_indicies = v_init<feature_index>();
        copy_array(tmp_f2_indicies, f2.indicies);
        for (size_t i1 = 0; i1 < f1.indicies.size(); i1++){
            size_t i2 = 0;
            for (i2 = 0; i2 < f2.indicies.size(); i2++){
                if (f1.indicies[i1] == f2.indicies[i2]){
                    f1.values[i1] = f1.values[i1]*f2.values[i2]/pow(norm_sq1*norm_sq2,0.5) - abs(f1.values[i1]/pow(norm_sq1,0.5) - f2.values[i2]/pow(norm_sq2,0.5));
                    //f1.values[i1] = f1.values[i1]*f2.values[i2] / pow(norm_sq1*norm_sq2, 0.5);
                    total_sum_feat_sq += pow(f1.values[i1],2);
                    tmp_f2_indicies[i2] = 0;
                    break;
                }
            }
            if (i2 == f2.indicies.size()){ //f1's index does not appear in f2, namely value of the index in f2 is zero.
                //f1.values[i1] = 0.0;
                f1.values[i1] = 0.0 - abs(f1.values[i1]/pow(norm_sq1,0.5));
                total_sum_feat_sq += pow(f1.values[i1],2);
            }
        }
        
        
        for (size_t i2 = 0; i2 < tmp_f2_indicies.size(); i2++){
            if (tmp_f2_indicies[i2] != 0){
                float value = 0.0 - abs(f2.values[i2]/pow(norm_sq2,0.5));
                f1.push_back(value, f2.indicies[i2]);
                total_sum_feat_sq += pow(value, 2);
            }
        }
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
    }*/

    void kronecker_product_f(features& f1, features& f2, features& f, float& total_sq, size_t& num_feat, uint64_t mask, size_t ss)
    {
        for (size_t i = 0; i < f1.indicies.size(); i++){
            size_t j = 0;
            for (j = 0; j < f2.indicies.size(); j++){
                if (f1.indicies[i] == f2.indicies[j]){    // != 0
                    f.push_back(f1.values[i]*f2.values[j], ((f1.indicies[i]+f2.indicies[j])<<ss) & mask);
                    total_sq += pow(f1.values[i]*f2.values[j], 2);
                    num_feat ++;
                    break;
                }
            }
        }
    }

    //void kronecker_product(example& ec1, example& ec2, example& ec, uint64_t mask, size_t ss)
    //{
        //copy_example_data(&ec, &ec1);
    //    ec.indices.push_back(node_id_namespace);
    //    ec.total_sum_feat_sq = 0.0;
    //    for (namespace_index c1 : ec1.indices){
    //        for (namespace_index c2 : ec2.indices){
    //            if (c1 == c2)
    //                kronecker_product_f(ec1.feature_space[c1], ec2.feature_space[c2], ec.feature_space[node_id_namespace], ec.total_sum_feat_sq, ec.num_features, mask, ss);
    //        }
    //    }
    //}

    void kronecker_product(example& ec1, example& ec2, example& ec)
    {
        copy_example_data(&ec, &ec1, true); 
        ec.indices.delete_v();
        //ec.indices.push_back(conditioning_namespace); //134, x86
        //ec.indices.push_back(dictionary_namespace); //135 x87
        unsigned char namespace_1 = 'x';
        unsigned char namespace_2 = 'y';
        ec.indices.push_back(namespace_1); //134, x86
        ec.indices.push_back(namespace_2); //135 x87
        ec.num_features = 0;
        ec.total_sum_feat_sq = 0.0;

        //to do: figure out how to set the indicies correctly (different namespaces may have same index)
        for (auto nc : ec1.indices)
        {
            for (size_t i = 0; i < ec1.feature_space[nc].indicies.size(); i++){
                //ec.feature_space[conditioning_namespace].push_back(ec1.feature_space[nc].values[i], ec1.feature_space[nc].indicies[i]);
                ec.feature_space[namespace_1].push_back(ec1.feature_space[nc].values[i], ec1.feature_space[nc].indicies[i]);
                ec.num_features++;
                ec.total_sum_feat_sq+=pow(ec1.feature_space[nc].values[i],2);
            }
        }
        for (auto nc : ec2.indices)
        {
            for (size_t i = 0; i < ec2.feature_space[nc].indicies.size(); i++){
                //ec.feature_space[dictionary_namespace].push_back(ec2.feature_space[nc].values[i], ec2.feature_space[nc].indicies[i]);
                ec.feature_space[namespace_2].push_back(ec2.feature_space[nc].values[i], ec2.feature_space[nc].indicies[i]);
                ec.num_features++;
                ec.total_sum_feat_sq+=pow(ec2.feature_space[nc].values[i],2);
            }
        }
    }

    ////////////////////////////end of helper/////////////////////////
    //////////////////////////////////////////////////////////////////


    ////////////////////////Implementation of memory_tree///////////////////
    ///////////////////////////////////////////////////////////////////////

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
        
        v_array<uint32_t> examples_index;

        node () //construct:
        {
            parent = 0;
            internal = 0; //0:not used, 1:internal, -1:leaf 
            //internal = false;
            depth = 0;
            base_router = 0;
            left = 0;
            right = 0;
            nl = 0.001; //initilze to 1, as we need to do nl/nr.
            nr = 0.001; 
            examples_index = v_init<uint32_t>();
        }
    };

    struct score_label
    {
        uint32_t label;
        float score;
        score_label()
        {
            label = 0;
            score = 0;
        }
        score_label(uint32_t l, float s){
            label = l;
            score = s;
        }
    };

    //memory_tree
    struct memory_tree
    {
        vw* all;

        v_array<node> nodes;  //array of nodes.
        v_array<example*> examples; //array of example points
        
        size_t max_leaf_examples;
        uint32_t num_queries; 
        size_t max_nodes;
        size_t max_routers;
        float alpha; //for cpt type of update.
        size_t routers_used;
        int iter;

        uint32_t total_num_queires; 

        size_t max_depth;
        size_t max_ex_in_leaf;

        float construct_time;
        float test_time;
        float cumulative_reward;
        float cumulative_reward_count;

        bool path_id_feat;
        bool hal_version;

        uint32_t num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;
        bool learn_at_leaf;

        bool test_mode;

        memory_tree()
        {
            nodes = v_init<node>();
            examples = v_init<example*>();
            alpha = 0.5;
            routers_used = 0;
            iter = 0;
            num_mistakes = 0;
            num_ecs = 0;
            num_test_ecs = 0;
            path_id_feat = false;
            test_mode = false;
            max_depth = 0;
            max_ex_in_leaf = 0;
            construct_time = 0;
            test_time = 0;
        }
    };

    float linear_kernel(const flat_example* fec1, const flat_example* fec2)
    { float dotprod = 0;
   
     features& fs_1 = (features&)fec1->fs;
     features& fs_2 = (features&)fec2->fs;
     if (fs_2.indicies.size() == 0)
       return 0.f;
   
     int numint = 0;
     for (size_t idx1 = 0, idx2 = 0; idx1 < fs_1.size() && idx2 < fs_2.size() ; idx1++)
     { uint64_t ec1pos = fs_1.indicies[idx1];
       uint64_t ec2pos = fs_2.indicies[idx2];
       //params.all->trace_message<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<<endl;
       if(ec1pos < ec2pos) continue;
   
       while(ec1pos > ec2pos && ++idx2 < fs_2.size())
         ec2pos = fs_2.indicies[idx2];
   
       if(ec1pos == ec2pos)
       { //params.all->trace_message<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<<endl;
         numint++;
         dotprod += fs_1.values[idx1] * fs_2.values[idx2];
         ++idx2;
       }
     }
     return dotprod;
   }

    float compute_l2_distance(memory_tree& b, example* ec1, example* ec2)
    {
        flat_example* fec1 = flatten_sort_example(*b.all, ec1);
        flat_example* fec2 = flatten_sort_example(*b.all, ec2);
        float linear_prod = linear_kernel(fec1, fec2);
        float l2_dis = 1. + 1. - 2.*linear_prod/pow(fec1->total_sum_feat_sq*fec2->total_sum_feat_sq, 0.5);
        fec1->fs.delete_v(); 
        fec2->fs.delete_v();
        free(fec1);
        free(fec2);
        return l2_dis;
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
        srand48(4000);
        //simple initilization: initilize the root only
        b.routers_used = 0;
        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.nodes[0].base_router = (b.routers_used++);

        b.total_num_queires = 0;
        b.max_routers = b.max_nodes;
        cout<<"tree initiazliation is done...."<<endl
            <<"max nodes "<<b.max_nodes<<endl
            <<"tree size: "<<b.nodes.size()<<endl
            <<"learn at leaf: "<<b.learn_at_leaf<<endl
            <<"num_queries per example: "<<b.num_queries<<endl;
    }


    //rout based on the prediction
    inline uint32_t descent(node& n, const float prediction)
    { 
        //prediction <0 go left, otherwise go right
        if(prediction < 0)
        {
            n.nl++; //increment the number of examples routed to the left.
            return n.left;
        }
        else //otherwise go right.
        {
            n.nr++; //increment the number of examples routed to the right.
            return n.right; 
        }
    }

    //return the id of the example and the leaf id (stored in cn)
    inline int random_sample_example_pop(memory_tree& b, uint32_t& cn)
    {
        cn = 0; //always start from the root:
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
                b.nodes[cn].nl--;
                cn = b.nodes[cn].left; 
            }
            else{
                b.nodes[cn].nr--;
                cn = b.nodes[cn].right;
            }
        }

        if (b.nodes[cn].examples_index.size() >= 1){
            int loc_at_leaf = int(merand48(b.all->random_state)*b.nodes[cn].examples_index.size());
	        uint32_t ec_id = b.nodes[cn].examples_index[loc_at_leaf];
            pop_at_index(b.nodes[cn].examples_index, loc_at_leaf); 
            return ec_id;
        }
        else    
            return -1;
    }


    //train the node with id cn, using the statistics stored in the node to
    //formulate a binary classificaiton example.
    float train_node(memory_tree& b, base_learner& base, example& ec, const uint32_t cn)
    {
        //predict, learn and predict
        //note: here we first train the router and then predict.
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        ec.l.simple = {1.f, 1.f, 0.};
        base.predict(ec, b.nodes[cn].base_router);
        float prediction = ec.pred.scalar; 
	//float imp_weight = 1.f; //no importance weight.
        
        float weighted_value = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr)/log(2.)+b.alpha*prediction;
        float route_label = weighted_value < 0.f ? -1.f : 1.f;
        
        //ec.l.simple = {route_label, imp_weight, 0.f}; 
	    ec.l.simple = {route_label, abs(weighted_value), 0.f};
        base.learn(ec, b.nodes[cn].base_router); //update the router according to the new example.
        
        base.predict(ec, b.nodes[cn].base_router);
        float save_binary_scalar = ec.pred.scalar;
        ec.l.multi = mc;
        ec.pred.multiclass = save_multi_pred;

        return save_binary_scalar;
    }

    
    //turn a leaf into an internal node, and create two children
    //when the number of examples is too big
    void split_leaf(memory_tree& b, base_learner& base, const uint32_t cn)
    {
        //create two children
        b.nodes[cn].internal = 1; //swith to internal node.
        uint32_t left_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());
        b.nodes[left_child].internal = -1;  //left leaf
        b.nodes[left_child].base_router = (b.routers_used++);
        uint32_t right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());  
        b.nodes[right_child].internal = -1;  //right leaf
        b.nodes[right_child].base_router = (b.routers_used++); 

        if (b.nodes[cn].depth + 1 > b.max_depth){
            b.max_depth = b.nodes[cn].depth + 1;
            cout<<"depth "<<b.max_depth<<endl;
        }

        b.nodes[cn].left = left_child;
        b.nodes[cn].right = right_child;
        b.nodes[left_child].parent = cn;
        b.nodes[right_child].parent = cn;
        b.nodes[left_child].depth = b.nodes[cn].depth + 1;
        b.nodes[right_child].depth = b.nodes[cn].depth + 1;

        if (b.nodes[left_child].depth > b.max_depth)
            b.max_depth = b.nodes[left_child].depth;

        //rout the examples stored in the node to the left and right
        for(size_t ec_id = 0; ec_id < b.nodes[cn].examples_index.size(); ec_id++) //scan all examples stored in the cn
        {
            uint32_t ec_pos = b.nodes[cn].examples_index[ec_id];
            MULTICLASS::label_t mc = b.examples[ec_pos]->l.multi;
            uint32_t save_multi_pred = b.examples[ec_pos]->pred.multiclass;
            b.examples[ec_pos]->l.simple = {1.f, 1.f, 0.f};
            base.predict(*b.examples[ec_pos], b.nodes[cn].base_router); //re-predict
            float scalar = b.examples[ec_pos]->pred.scalar;
            if (scalar < 0)
            {
                b.nodes[left_child].examples_index.push_back(ec_pos);
                float leaf_pred = train_node(b, base, *b.examples[ec_pos], left_child);
                descent(b.nodes[left_child], leaf_pred); //fake descent, only for update nl and nr                
            }
            else
            {
                b.nodes[right_child].examples_index.push_back(ec_pos);
                float leaf_pred = train_node(b, base, *b.examples[ec_pos], right_child);
                descent(b.nodes[right_child], leaf_pred); //fake descent. for update nr and nl
            }
            b.examples[ec_pos]->l.multi = mc;
            b.examples[ec_pos]->pred.multiclass = save_multi_pred;
        }
        b.nodes[cn].examples_index.delete_v(); //empty the cn's example list
        b.nodes[cn].nl = std::max(double(b.nodes[left_child].examples_index.size()), 0.001); //avoid to set nl to zero
        b.nodes[cn].nr = std::max(double(b.nodes[right_child].examples_index.size()), 0.001); //avoid to set nr to zero

        if (std::max(b.nodes[cn].nl, b.nodes[cn].nr) > b.max_ex_in_leaf)
        {
            b.max_ex_in_leaf = std::max(b.nodes[cn].nl, b.nodes[cn].nr);
            //cout<<b.max_ex_in_leaf<<endl;
        }
        
    }
    
    //add path feature:
    void add_node_id_feature (memory_tree& b, uint32_t cn, example& ec)
    {
        vw* all = b.all;
        uint64_t mask = all->weights.mask();
        size_t ss = all->weights.stride_shift();

        ec.indices.push_back (node_id_namespace);
        features& fs = ec.feature_space[node_id_namespace];

        while (cn > 0)
        { 
            fs.push_back (1., ((868771 * cn) << ss) & mask);
            cn = b.nodes[cn].parent;
        }
    }

    uint32_t majority_vote(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        if(b.nodes[cn].examples_index.size() > 0){
            v_array<score_label> score_labs = v_init<score_label>();
            for(size_t i = 0; i < b.nodes[cn].examples_index.size(); i++){
                float score = 0.f;
                uint32_t loc = b.nodes[cn].examples_index[i];
                if (b.learn_at_leaf == true){
                    //cout<<"learn at leaf"<<endl;
                    example* kprod_ec = &calloc_or_throw<example>();
                    diag_kronecker_product_test(ec, *b.examples[loc], *kprod_ec);
                    kprod_ec->l.simple = {-1., 1., 0.};
                    base.predict(*kprod_ec, b.max_routers);
                    //score = kprod_ec->pred.scalar;
                    score = kprod_ec->partial_prediction;
                    free_example(kprod_ec);
                }
                else
                    score = normalized_linear_prod(b, &ec, b.examples[loc]);
                
                bool in = false;
                for (score_label& s_l : score_labs){
                    if (s_l.label == b.examples[loc]->l.multi.label ){
                        s_l.score += exp(50.*(score));
                        in = true;
                        break;
                    }
                }
                if (in == false)
                    score_labs.push_back(score_label(b.examples[loc]->l.multi.label, score));   
            }
            float max_score = -FLT_MAX;
            uint32_t max_lab = 0;
            for (size_t j = 0; j < score_labs.size(); j++){
                if (score_labs[j].score > max_score)
                {
                    max_score = score_labs[j].score;
                    max_lab = score_labs[j].label;
                }
            }
            return max_lab;
        }
        else
            return 0;
    }




    //pick up the "closest" example in the leaf using the score function.
    int64_t pick_nearest(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        if (b.nodes[cn].examples_index.size() > 0)
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
                    //tmp_s = 0;
                    example* kprod_ec = &calloc_or_throw<example>();
                    diag_kronecker_product_test(ec, *b.examples[loc], *kprod_ec);
                    kprod_ec->l.simple = {FLT_MAX, 1., tmp_s};
                    base.predict(*kprod_ec, b.max_routers);
                    //score = kprod_ec->pred.scalar;
                    score = kprod_ec->partial_prediction;
                    free_example(kprod_ec);
                }
                else
                    //score = -1.*compute_l2_distance(b, &ec, b.examples[loc]); 
                    score = normalized_linear_prod(b, &ec, b.examples[loc]);
                
                if (score > max_score){
                    max_score = score;
                    max_pos = (int64_t)loc;
                }
            }
            return max_pos;
        }
        else
            return -1;
    }

    void learn_similarity_at_leaf(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        for (uint32_t loc : b.nodes[cn].examples_index)
        {
            example* ec_loc = b.examples[loc];
            //float score = -compute_l2_distance(b, &ec, ec_loc); //score based on the l2_distance. 
            float score = normalized_linear_prod(b, &ec, ec_loc);
            //score = 0.0;
            example* kprod_ec = &calloc_or_throw<example>();
            diag_kronecker_product_test(ec, *ec_loc, *kprod_ec);
            //
            if (ec.l.multi.label == b.examples[loc]->l.multi.label) //reward = 1:    
                kprod_ec->l.simple = {1., 1., -score};
            else
                //kprod_ec->l.simple = {-1., 1., score}; //reward = 0:
                kprod_ec->l.simple = {-0., 1., -score};
        
            base.learn(*kprod_ec, b.max_routers);
            free_example(kprod_ec);
        }
    }

    void split_leaf_hal(memory_tree& b, base_learner& base, const uint32_t cn)
    {
        //create two children
        b.nodes[cn].internal = 1; //swith to internal node.
        uint32_t left_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());
        b.nodes[left_child].internal = -1;  //left leaf
        b.nodes[left_child].base_router = (b.routers_used++);
        uint32_t right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());  
        b.nodes[right_child].internal = -1;  //right leaf
        b.nodes[right_child].base_router = (b.routers_used++); 

        if (b.nodes[cn].depth + 1 > b.max_depth){
            b.max_depth = b.nodes[cn].depth + 1;
            //cout<<"depth "<<b.max_depth<<endl;
        }

        b.nodes[cn].left = left_child;
        b.nodes[cn].right = right_child;
        b.nodes[left_child].parent = cn;
        b.nodes[right_child].parent = cn;
        b.nodes[left_child].depth = b.nodes[cn].depth + 1;
        b.nodes[right_child].depth = b.nodes[cn].depth + 1;

        for(size_t ec_id = 0; ec_id < b.nodes[cn].examples_index.size(); ec_id++) //scan all examples stored in the cn
        {
            float left_reward = 0.;
            float right_reward = 0.;
            uint32_t ec_pos = b.nodes[cn].examples_index[ec_id];
            int64_t left_closest = pick_nearest(b,base,left_child, *b.examples[ec_pos]);
            int64_t right_closest= pick_nearest(b,base,right_child,*b.examples[ec_pos]);
            if ((left_closest != -1) && (b.examples[left_closest]->l.multi.label == b.examples[ec_pos]->l.multi.label)) {
                left_reward = b.examples[ec_pos]->weight;
            }
            if ((right_closest != -1) && (b.examples[right_closest]->l.multi.label == b.examples[ec_pos]->l.multi.label)) {
                right_reward = b.examples[ec_pos]->weight;
            }
            float objective = (1-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(right_reward - left_reward);

            MULTICLASS::label_t mc = b.examples[ec_pos]->l.multi;
            uint32_t save_multi_pred = b.examples[ec_pos]->pred.multiclass;
            b.examples[ec_pos]->l.simple = {objective < 0. ? -1.f:1.f, fabs(objective), 0.};
            base.learn(*b.examples[ec_pos], b.nodes[cn].base_router);
            base.predict(*b.examples[ec_pos], b.nodes[cn].base_router);
            float scalar = b.examples[ec_pos]->pred.scalar;
            if (scalar < 0){
                b.nodes[left_child].examples_index.push_back(ec_pos);
                b.nodes[cn].nl ++;
            }
            else{
                b.nodes[right_child].examples_index.push_back(ec_pos);
                b.nodes[cn].nr ++;
            }
            b.examples[ec_pos]->l.multi = mc;
            b.examples[ec_pos]->pred.multiclass = save_multi_pred;
        }
        b.nodes[cn].examples_index.delete_v(); 
        b.nodes[cn].nl = std::max(double(b.nodes[left_child].examples_index.size()), 0.001);
        b.nodes[cn].nr = std::max(double(b.nodes[right_child].examples_index.size()), 0.001); //avoid to set nr to zero
        
        if (std::max(b.nodes[cn].nl, b.nodes[cn].nr) > b.max_ex_in_leaf)
        {
            b.max_ex_in_leaf = std::max(b.nodes[cn].nl, b.nodes[cn].nr);
            //cout<<"max example in leaf now: "<<b.max_ex_in_leaf<<endl;
        }
    }


    void predict(memory_tree& b, base_learner& base, example& test_ec)
    {
        example& ec = calloc_or_throw<example>();
        //ec = *b.examples[b.iter];
        copy_example_data(&ec, &test_ec);
        remove_repeat_features_in_ec(ec);
        
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        uint32_t cn = 0;
        ec.l.simple = {-1.f, 1.f, 0.};
        while(b.nodes[cn].internal == 1) //if it's internal
        {
            //cout<<"at node "<<cn<<endl;
            base.predict(ec, b.nodes[cn].base_router);
            uint32_t newcn = ec.pred.scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right; //do not need to increment nl and nr.
            cn = newcn;
        }
         
        ec.l.multi = mc; 
        ec.pred.multiclass = save_multi_pred;

        
        /*
        uint32_t label = majority_vote(b, base, cn, ec);
        if (label != 0)
        {
            ec.pred.multiclass = label;
            test_ec.pred.multiclass = label;
            if(test_ec.pred.multiclass != test_ec.l.multi.label){
                test_ec.loss = test_ec.weight; //weight in default is 1.
                b.num_mistakes++;
            }
        }
        else{
            test_ec.loss = test_ec.weight;
            b.num_mistakes++;
        }*/

        int64_t closest_ec = pick_nearest(b, base, cn, ec);
        if (closest_ec != -1){
            ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            test_ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            if(test_ec.pred.multiclass != test_ec.l.multi.label)
            {
                test_ec.loss = test_ec.weight; //weight in default is 1.
                b.num_mistakes++;
            }
        }
        else{
            test_ec.loss = test_ec.weight;
            b.num_mistakes++;
        }

        free_example(&ec);
    }



  float insert_example_hal_helper(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, uint32_t cn, bool deviated, bool insert_only) {
    //cerr << "insert_example_hal_helper(" << cn << ")" << endl;
    example& ec = *b.examples[ec_array_index];
    // base case: leaf
    if (b.nodes[cn].internal == -1) { // got to a leaf
        if (insert_only) {
	    //cerr << "insert_only=1 cn=" << cn << endl;
	        b.nodes[cn].examples_index.push_back(ec_array_index); // do the insert
	        if (b.learn_at_leaf == true)
	            learn_similarity_at_leaf(b, base, cn, ec);
	        return 0.;
        }
        // get the reward
        float final_reward = 0.;
        int64_t closest_ec = pick_nearest(b, base, cn, ec);
        if ((closest_ec != -1) && (b.examples[closest_ec]->l.multi.label == ec.l.multi.label))
	        final_reward = ec.weight;
      
        base.predict(ec, b.nodes[cn].base_router);
        float prediction = ec.pred.scalar;
        descent(b.nodes[cn], prediction); // faked descent
        if((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size()+2 < b.max_nodes))
	        split_leaf_hal(b, base, cn);

        //cerr << "insert_only=0 cn=" << cn << " deviated=" << deviated << " final_reward=" << final_reward << " closest_ec=" << closest_ec << endl;
        return final_reward;
    }

    // inductive case: internal node
    float p_follow = (false && deviated) ? 1. : (1. - 1. / (float)(b.max_depth));
    if (insert_only) p_follow = 1.; 
    // get the current node's prediction
    base.predict(ec, b.nodes[cn].base_router);
    float prediction = ec.pred.scalar;
    bool deviate = merand48(b.all->random_state) > p_follow;
    if (deviate) prediction = -prediction;
    uint32_t newcn = descent(b.nodes[cn], prediction);
    // recurse
    float final_reward = insert_example_hal_helper(b, base, ec_array_index, newcn, deviated || deviate, insert_only);
    // consider using a control variate baseline
    float baseline = 0.; // (b.cumulative_reward_count < 1) ? 0. : (b.cumulative_reward / b.cumulative_reward_count);
    float ips_reward = (final_reward - baseline) / (deviate ? (1. - p_follow) : p_follow); // IPS on the reward
    // if reward is positive, that should push us in the direction of _prediction_
    if (prediction < 0) ips_reward = -ips_reward;
    // update this node's router
    float objective = (1. - b.alpha) * log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha * ips_reward/2.;
    //if ((!insert_only) && (merand48(b.all->random_state) < 1e-4))
    //  cerr << "final_reward=" << final_reward << "\tdeviate=" << deviate << "\tp_follow=" << p_follow << "\tips_reward=" << ips_reward << "\tobjective=" << objective << endl;
    // update
    MULTICLASS::label_t mc = ec.l.multi;
    ec.l.simple = { objective < 0. ? -1.f : 1.f, fabs(objective) , 0. };
    base.learn(ec, b.nodes[cn].base_router);
    ec.l.multi = mc; // restore label

    return final_reward;
  }

    float return_reward_from_node(memory_tree& b, base_learner& base, uint32_t cn, example& ec){
        //example& ec = *b.examples[ec_array_index];
        while(b.nodes[cn].internal != -1){
            base.predict(ec, b.nodes[cn].base_router);
            float prediction = ec.pred.scalar;
            cn = prediction < 0 ? b.nodes[cn].left : b.nodes[cn].right;
        }
        //get to leaf now:
        int64_t closest_ec = pick_nearest(b,base,cn, ec);  //location at b.examples.
        float reward = 0.;
        if ((closest_ec != -1) && (b.examples[closest_ec]->l.multi.label == ec.l.multi.label))
            reward = 1.;
        b.total_num_queires ++;

        if (b.learn_at_leaf == true){
            //float score = normalized_linear_prod(b, &ec, b.examples[closest_ec]);
            example* kprod_ec = &calloc_or_throw<example>();
            diag_kronecker_product_test(ec, *b.examples[closest_ec], *kprod_ec);
            kprod_ec->l.simple = {reward, 1.f, 0.f};
            base.learn(*kprod_ec, b.max_routers);
            free_example(kprod_ec);
        }
        return reward;
    }


    float query_reward(memory_tree& b, base_learner& base, const uint32_t& leaf_id, example& ec){
        b.total_num_queires ++;
        
        float reward = 0.f;
        int64_t closest_ec = pick_nearest(b, base, leaf_id, ec);
		if ((closest_ec != -1) && (b.examples[closest_ec]->l.multi.label == ec.l.multi.label))
			reward = 1.; //ec.weight
        
        if (b.learn_at_leaf == true){
            //float score = normalized_linear_prod(b, &ec, b.examples[closest_ec]);
            example* kprod_ec = &calloc_or_throw<example>();
            diag_kronecker_product_test(ec, *b.examples[closest_ec], *kprod_ec);
            kprod_ec->l.simple = {reward, 1.f, 0.f};
            base.learn(*kprod_ec, b.max_routers);
            free_example(kprod_ec);
        }
        return reward;
    }


    void route_to_leaf(memory_tree& b, base_learner& base, const uint32_t & ec_array_index, uint32_t cn, v_array<uint32_t>& path, bool insertion){
		example& ec = *b.examples[ec_array_index];
        path.erase();
		while(b.nodes[cn].internal != -1){
			path.push_back(cn);  //path stores node id from the root to the leaf
			base.predict(ec, b.nodes[cn].base_router);
			float prediction = ec.pred.scalar;
			if (insertion == false)
				cn = prediction < 0 ? b.nodes[cn].left : b.nodes[cn].right;
			else
			    cn = descent(b.nodes[cn], prediction);
		}
		path.push_back(cn); //push back the leaf 
        //cout<<"at route to leaf: "<<path.size()<<endl;
		if (insertion == true){
			b.nodes[cn].examples_index.push_back(ec_array_index);
			if ((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 < b.max_nodes))
				split_leaf(b,base,cn);
		}
	}

    void query_and_learn(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, const uint32_t& K)
    {
        v_array<uint32_t> path_to_leaf = v_init<uint32_t>();
		example& ec = *b.examples[ec_array_index];
		route_to_leaf(b, base, ec_array_index, 0, path_to_leaf, true);
        uint32_t leaf_id = path_to_leaf[path_to_leaf.size()-1];
        //cout<<"at query and learn 1: "<<path_to_leaf.size()<<endl;

        if (path_to_leaf.size() > 1){
            float reward_from_path = query_reward(b, base, leaf_id, ec);

            v_array<uint32_t> sample_pos = v_init<uint32_t>();
            uint32_t query_times = (std::min) (uint32_t(path_to_leaf.size()-1), K);
            //cout<<path_to_leaf.size()<<" "<<K<<" "<<query_times<<endl;
            for (uint32_t i = 0; i < path_to_leaf.size() - 1; i++){ //go through the nodes in path, with probablity query_times/total_length, chose this node. hence, in average, we chose query_times nodes for explore
                bool choosen = merand48(b.all->random_state) < (query_times*1.f /(path_to_leaf.size()*1.f - 1.f));
                if (choosen)
                    sample_pos.push_back(i);
            }
            //explore at each choosen node:
            //cout<<"query times at this example: "<<sample_pos.size()<<endl;
            //for (auto pos : sample_pos)
            //    cout<<path_to_leaf[pos]<<" ";
            //cout<<endl;

            for (auto pos : sample_pos){
                uint32_t cn = path_to_leaf[pos];
                base.predict(ec, b.nodes[cn].base_router);
                float prediction = ec.pred.scalar;
                float p_follow = (std::max)(0.f, (std::min)(1.f, 0.5f*(1.f+2.f*fabs(prediction)))); //prob of follow ranging from (0.5, 0.99)
			    bool deviate = merand48(b.all->random_state) > p_follow;
			    if (deviate) prediction = -prediction; //flip the sign of prediction is deviate is true
                
                float final_reward = 0.f;
                if (deviate){
                    uint32_t deviated_to_node_id = prediction < 0 ? b.nodes[cn].left : b.nodes[cn].right;
                    v_array<uint32_t> deivated_path = v_init<uint32_t>();//query new path.
				    route_to_leaf(b, base, ec_array_index, deviated_to_node_id, deivated_path, false);
                    //cout<<"at query and learn 2: "<<deivated_path.size()<<endl;
				    uint32_t deviated_leaf_id = deivated_path[deivated_path.size() - 1]; //get the leaf from the deviated path
				    deivated_path.delete_v(); //free memory
                    final_reward = query_reward(b, base, deviated_leaf_id, ec);  //query happens here
                }
                else
                    final_reward = reward_from_path;

                float ips_reward = final_reward / (deviate ? (1. - p_follow) : p_follow); // IPS on the reward
			    if (prediction < 0) ips_reward = -ips_reward;
			    float objective = (1. - b.alpha) * log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha * ips_reward/2.;
			    MULTICLASS::label_t mc = ec.l.multi;
			    ec.l.simple = { objective < 0. ? -1.f : 1.f, fabs(objective) , 0. };
                base.learn(ec, b.nodes[cn].base_router);
                ec.l.multi = mc;
            }
            sample_pos.delete_v();
        }
        //once we finish updating, we officially insert the example to the memory. 
        //route_to_leaf(b, base, ec_array_index, 0, path_to_leaf, true); //insert
        path_to_leaf.delete_v();
    }

    void insert_example_without_ips(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, bool insert_only)
    {
        example& ec = *b.examples[ec_array_index];
        uint32_t cn = 0;

        while(b.nodes[cn].internal != -1) //internal nodes
        {
            float prediction = 0.;
            //if (!insert_only) //do train the node first: 
            bool train_flag = (merand48(b.all->random_state) <= (b.num_queries*1.f/(b.max_depth*1.f)));
            //cout<<b.num_queries<<" "<<" "<<b.max_depth<<" "<<train_flag<<endl;
            if (train_flag == true){
            //if (true){
                //randomize it:
                base.predict(ec, b.nodes[cn].base_router);
                float curr_pred = ec.pred.scalar;
                float prob_right = (std::max)(0.0f, (std::min) (1.f, 0.5f * (1.f + 2.f*curr_pred)));
                prob_right = 0.9f*prob_right + 0.1f/2.f;
                float coin = merand48(b.all->random_state) < prob_right ? 1.f : -1.f;
                
                float objective = 0.f;
                if (coin == -1.f){ //go left
                    float reward_left_subtree = return_reward_from_node(b,base, b.nodes[cn].left, ec);
                    objective = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(-reward_left_subtree/(1.-prob_right))/2.;
                }
                else{ //go right:
                    float reward_right_subtree= return_reward_from_node(b,base, b.nodes[cn].right, ec);
                    objective = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(reward_right_subtree/prob_right)/2.;
                }

                float ec_input_weight = ec.weight;
                MULTICLASS::label_t mc = ec.l.multi;
                //ec.l.simple.label = (objective < 0. ? -1.f : 1.f);

                ec.weight = fabs(objective);
                if (ec.weight >= 100.f)
                    ec.weight = 100.f;
                else if (ec.weight < .01f)
                    ec.weight = 0.01f;
                    
                //ec.weight = (fabs(objective) <= 0.001 ? 1 : fabs(objective)); 
                ec.l.simple = {objective < 0. ? -1.f : 1.f, 1.f, 0.};
                base.learn(ec, b.nodes[cn].base_router);
                ec.l.multi = mc;
                ec.weight = ec_input_weight; //restore the original weight
            }
            base.predict(ec, b.nodes[cn].base_router);
            prediction = ec.pred.scalar;
            cn = descent(b.nodes[cn], prediction);
        }
        //now we get to leaf: we only train the leaf predictor at the phase of insertion.
        //we only split the node at the phase of insertion. (we do not need to split leaf at learn phase, as there won't be any new insertion.)
        if (insert_only == true){
            b.nodes[cn].examples_index.push_back(ec_array_index); //insert to the leaf.
            //if (b.learn_at_leaf == true)
            //    learn_similarity_at_leaf(b,base,cn, ec);      
            if ((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 < b.max_nodes))
                split_leaf_hal(b,base,cn);
        }
    }

    void insert_example_hal(memory_tree& b, base_learner& base, const uint32_t& ec_array_index) 
    {
        //insert_example_hal_helper(b, base, ec_array_index, 0, false, true); // insert only
        insert_example_without_ips(b, base, ec_array_index, true);
        //query_and_learn(b, base, ec_array_index, b.num_queries);
        
        //float final_reward = insert_example_hal_helper(b, base, ec_array_index, 0, false, false); // learn
        //insert_example_without_ips(b, base, ec_array_index, false);
        //b.cumulative_reward += final_reward;
        //b.cumulative_reward_count += 1.;
    }

    //node here the ec is already stored in the b.examples, the task here is to rout it to the leaf, 
    //and insert the ec_array_index to the leaf.
    void insert_example(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, bool fake_insert = false)
    {
        uint32_t cn = 0; //start from the root.
        while(b.nodes[cn].internal == 1) //if it's internal node:
        {   
            //predict and train the node at cn.
            float router_pred = train_node(b, base, *b.examples[ec_array_index], cn); 
            uint32_t newcn = descent(b.nodes[cn], router_pred); //updated nr or nl
            cn = newcn; 
        }
        //at leaf, use available information at leaf to train regressor/classifier at leaf.
        if ((b.nodes[cn].internal == -1) && (b.learn_at_leaf == true))
            learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]);

        if((b.nodes[cn].internal == -1) && (fake_insert == false)) //get to leaf:
        {   
            b.nodes[cn].examples_index.push_back(ec_array_index);
            if (b.nodes[cn].examples_index.size() > b.max_ex_in_leaf)
            {
                b.max_ex_in_leaf = b.nodes[cn].examples_index.size();
                //cout<<cn<<" "<<b.max_ex_in_leaf<<endl;
            }
            float leaf_pred = train_node(b, base, *b.examples[ec_array_index], cn); //tain the leaf as well.
            descent(b.nodes[cn], leaf_pred); //this is a faked descent, the purpose is only to update nl and nr of cn

            //if the number of examples exceeds the max_leaf_examples, and not reach the max_nodes - 2 yet, we split:
            if((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 <= b.max_nodes)){
                split_leaf(b, base, cn); 
            }
            //else{
            //    if (b.learn_at_leaf == true){
	        //        learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]);  //learn similarity function at leafb.l
            //    }
            //}
        }
        //else if((b.nodes[cn].internal == -1) && (fake_insert == true))
        //    learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]); 
    }

    void experience_replay(memory_tree& b, base_learner& base)
    {
      //return; // TODO turn this back on
        uint32_t cn = 0; //start from root, randomly descent down! 
        //int loc_at_leaf = random_sample_example_pop(b, cn);
	    int ec_id = random_sample_example_pop(b,cn);
        //if (loc_at_leaf >= 0){
	    if (ec_id >= 0){
            //uint32_t ec_id = b.nodes[cn].examples_index[loc_at_leaf]; //ec_id is the postion of the sampled example in b.examples. 
            //re-insert:note that we do not have to 
            //restore the example into b.examples, as it's alreay there
            //insert_example(b, base, ec_id); 
            //insert_example_hal_helper(b, base, ec_id, 0, false, true); // insert only
            //float final_reward = insert_example_hal_helper(b, base, ec_id, 0, false, false); // learn
            //insert_example_without_ips(b, base, ec_id, true);
            
            if (b.hal_version == true){
                v_array<uint32_t> tmp_path = v_init<uint32_t>();
                route_to_leaf(b, base, ec_id, 0, tmp_path, true);
                tmp_path.delete_v();
            }
            else
                insert_example(b, base, ec_id); 
        }

    }

    //learn: descent the example from the root while generating binary training
    //example for each node, including the leaf, and store the example at the leaf.
    void learn(memory_tree& b, base_learner& base, example& ec)
    {        
        if (b.test_mode == false){
            b.iter++;
            predict(b, base, ec);
            if (b.iter%5000 == 0)
	            cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<", total num queires so far: "<<b.total_num_queires<<", max depth: "<<b.max_depth<<", max exp in leaf: "<<b.max_ex_in_leaf<<endl;

            clock_t begin = clock();
            example* new_ec = &calloc_or_throw<example>();
            copy_example_data(new_ec, &ec);
            remove_repeat_features_in_ec(*new_ec); ////sort unique.
            b.examples.push_back(new_ec);   
            b.num_ecs++; 

            float random_prob = merand48(b.all->random_state);
            if (random_prob < 1.) {
	            if (b.hal_version){
		            insert_example_hal(b, base, b.examples.size()-1);
                }
	        else
		        insert_example(b, base, b.examples.size()-1);
	        }
            else{
	            if (b.hal_version) cerr << "eeeeeek!" << endl;
                insert_example(b, base, b.examples.size()-1, true);
                b.num_ecs--;
                free_example(new_ec);
                b.examples.pop();
            }
            //if (b.iter % 1 == 0)
            for (int i = 0; i < 1; i++)
                experience_replay(b, base);
            b.construct_time = double(clock() - begin)/CLOCKS_PER_SEC;   
        }
        else if (b.test_mode == true){
            b.iter++;
            if (b.iter % 5000 == 0)
                cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
            clock_t begin = clock();
            predict(b, base, ec);
            b.test_time += double(clock() - begin)/CLOCKS_PER_SEC;
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
        cout<<b.construct_time<<" "<<b.test_time<<endl;
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

base_learner* memory_tree_setup(vw& all)
{
    using namespace memory_tree_ns;
    if (missing_option<uint32_t, true>(all, "memory_tree", "Make a memory tree with at most <n> nodes"))
        return nullptr;
    
    new_options(all, "memory tree options")
      ("leaf_example_multiplier", po::value<uint32_t>()->default_value(1.0), "multiplier on examples per leaf (default = log nodes)")
      ("hal_version", po::value<bool>()->default_value(false), "use reward based optimization")
      ("learn_at_leaf", po::value<bool>()->default_value(true), "whether or not learn at leaf (defualt = True)")
      ("num_queries", po::value<uint32_t>()->default_value(1.), "number of returned queries per example (<= log nodes")
      ("Alpha", po::value<float>()->default_value(0.1), "Alpha");
     add_options(all);

    po::variables_map& vm = all.vm;
    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.all = &all;
    tree.max_nodes = vm["memory_tree"].as<uint32_t>();
    tree.learn_at_leaf = vm["learn_at_leaf"].as<bool>();

    if (vm.count("leaf_example_multiplier"))
      {
	tree.max_leaf_examples = vm["leaf_example_multiplier"].as<uint32_t>() * (log(tree.max_nodes)/log(2));
	*all.file_options << " --leaf_example_multiplier " << vm["leaf_example_multiplier"].as<uint32_t>();
      }

    if (vm.count("num_queries")){
        tree.num_queries = vm["num_queries"].as<uint32_t>(); 
        *all.file_options << " --num_queries "<< vm["num_queries"].as<uint32_t>();
    }

    if (vm.count("Alpha"))
      {
	tree.alpha = vm["Alpha"].as<float>();
	*all.file_options << " --Alpha " << tree.alpha;
      }
    if (vm.count("hal_version"))
      {
	tree.hal_version =  vm["hal_version"].as<bool>();  //vm.count("hal_version") > 0;
	*all.file_options << " --hal_version "<<tree.hal_version;
      }
    
    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "memory_tree:" << " "
                    <<"max_nodes = "<< tree.max_nodes << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
 		    <<"alpha = "<<tree.alpha<<" "
		    <<"hal_version = " << tree.hal_version<< " "
                    <<std::endl;
    
    learner<memory_tree>& l = 
        init_multiclass_learner (&tree, 
                setup_base (all),
                learn,
                predict,
                all.p, 
                tree.max_nodes + 1);
    
    //srand(time(0));
    l.set_save_load(save_load_memory_tree);
    l.set_finish(finish);

    return make_base (l);
}







//learning rate for aloi: 0.0001


/*
$ zcat aloi_train.vw.gz | sed 's/|/|f/' | ../../vowpalwabbit/vw -k -c --passes 8000 --memory_tree 1000 -b 26 -q'\x88': -l 0.0001 --Alpha 0.5 --early_terminate 9999 --holdout_off
average  since         example        example  current  current  current
loss     last          counter         weight    label  predict features
1.000000 1.000000            1            1.0      796        0       46
1.000000 1.000000            2            2.0      712      796       23
1.000000 1.000000            4            4.0      590      676       20
1.000000 1.000000            8            8.0      835      712       46
1.000000 1.000000           16           16.0      995      331       40
1.000000 1.000000           32           32.0      200      387       24
1.000000 1.000000           64           64.0      697      150       41
1.000000 1.000000          128          128.0      344      764       16
0.992188 0.984375          256          256.0       98       14       31
0.976562 0.960938          512          512.0       13        1       38
0.972656 0.968750         1024         1024.0      113      415       29
0.970703 0.968750         2048         2048.0      670      508       22
0.967529 0.964355         4096         4096.0      340      699       30
at iter 5000, pred error: 0.9658, baseline=-nan
0.955322 0.943115         8192         8192.0      426      485       61
at iter 10000, pred error: 0.9537, baseline=-nan
at iter 15000, pred error: 0.942467, baseline=-nan
0.938538 0.921753        16384        16384.0      972      300       44
at iter 20000, pred error: 0.9323, baseline=-nan
at iter 25000, pred error: 0.92568, baseline=-nan
at iter 30000, pred error: 0.918733, baseline=-nan
0.915924 0.893311        32768        32768.0      431      431       34
at iter 35000, pred error: 0.913114, baseline=-nan
at iter 40000, pred error: 0.90895, baseline=-nan
at iter 45000, pred error: 0.904756, baseline=-nan
at iter 50000, pred error: 0.90042, baseline=-nan
at iter 55000, pred error: 0.897218, baseline=-nan
at iter 60000, pred error: 0.893933, baseline=-nan
at iter 65000, pred error: 0.890892, baseline=-nan
0.890259 0.864594        65536        65536.0      645      759       22
at iter 70000, pred error: 0.8865, baseline=-nan
at iter 75000, pred error: 0.88352, baseline=-nan
at iter 80000, pred error: 0.879463, baseline=-nan
at iter 85000, pred error: 0.876694, baseline=-nan
at iter 90000, pred error: 0.874333, baseline=-nan
at iter 95000, pred error: 0.872189, baseline=-nan
at iter 100000, pred error: 0.87347, baseline=-nan
at iter 105000, pred error: 0.877048, baseline=-nan
at iter 110000, pred error: 0.880055, baseline=-nan
at iter 115000, pred error: 0.882809, baseline=-nan
at iter 120000, pred error: 0.8844, baseline=-nan
at iter 125000, pred error: 0.885464, baseline=-nan
at iter 130000, pred error: 0.8856, baseline=-nan
0.885818 0.881378       131072       131072.0      253      226       37
 */
