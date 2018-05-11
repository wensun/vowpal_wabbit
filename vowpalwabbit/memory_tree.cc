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
#include "bleu.h"


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


    float inner_product_two_features(const features& fs_1, const features& fs_2, double norm_sq1 = 1., double norm_sq2 = 1.)
    {
        float dotprod = 0.0;
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
        return (dotprod/pow(norm_sq1*norm_sq2, 0.5));
    }

    float linear_kernel(const flat_example* fec1, const flat_example* fec2)
    { 
        float dotprod = 0;
        features& fs_1 = (features&)fec1->fs;
        features& fs_2 = (features&)fec2->fs;
        dotprod = inner_product_two_features(fs_1,fs_2);
        return dotprod;
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

    void diag_kronecker_product_test(example& ec1, example& ec2, example& ec, uint32_t task_id = 1)
    {
        if (task_id == 1){
            copy_example_data(&ec, &ec1);
            ec.total_sum_feat_sq = 0.0;
            for(namespace_index c1 : ec1.indices){
                for(namespace_index c2 : ec2.indices){
                    if (c1 == c2)
                        diag_kronecker_prod_fs_test(ec1.feature_space[c1], ec2.feature_space[c2], ec.feature_space[c1], ec.total_sum_feat_sq, ec1.total_sum_feat_sq, ec2.total_sum_feat_sq);
                }
            }
        }
        else if (task_id != 1) //Q and A (J and I)
        {
            unsigned char ns = 'J';
            copy_example_data(&ec, &ec1, true);
            ec.indices.push_back(ns);
            ec.feature_space[ns].deep_copy_from(ec1.feature_space[ns]);
            double ec1_total_feat_sq = 0.;
            double ec2_total_feat_sq = 0.;
            for (size_t t = 0; t < ec1.feature_space[ns].values.size(); t++)
                ec1_total_feat_sq += pow(ec1.feature_space[ns].values[t],2);
            for (size_t t = 0; t < ec2.feature_space[ns].values.size(); t++)
                ec2_total_feat_sq += pow(ec2.feature_space[ns].values[t],2);

            diag_kronecker_prod_fs_test(ec1.feature_space[ns], ec2.feature_space[ns], ec.feature_space[ns], ec.total_sum_feat_sq, ec1_total_feat_sq, ec2_total_feat_sq);  
        }
    }

    inline float square_norm_feature(features& fs){
        double total_feat_sum_sq = 0.;
        for (size_t t = 0; t < fs.values.size(); t++)
            total_feat_sum_sq += pow(fs.values[t],2);
        return total_feat_sum_sq;
    }

    float compute_similarity_under_namespace(example& ec1, example& ec2, unsigned char ns)
    {
        features& f1 = ec1.feature_space[ns];
        features& f2 = ec2.feature_space[ns];
        //double f1_feat_sum_sq = square_norm_feature(f1);
        //double f2_feat_sum_sq = square_norm_feature(f2);
        //compute the innner product between these twos:
        float dotprod = inner_product_two_features(f1, f2, 1., 1.);
        //float dotprod = inner_product_two_features(f1, f2, f1_feat_sum_sq, f2_feat_sum_sq);
        //size_t d = f1.values.size();
        //float dotprod = 0.;
        //for (size_t i = 0; i < d; i++)
        //    dotprod += f1.values[i]*f2.values[i];
        
        return dotprod;
    }


    
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
    void diag_kronecker_product(example& ec1, example& ec2, example& ec, uint32_t task_id = 1)
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

    //memory_tree
    struct memory_tree
    {
        vw* all;

        v_array<node> nodes;  //array of nodes.
        v_array<example*> examples; //array of example points
        
        size_t max_leaf_examples; 
        uint32_t task_id;
        size_t train_N;
        size_t max_nodes;
        size_t max_routers;
        float alpha; //for cpt type of update.
        size_t routers_used;
        size_t iter;

	bool bandit;
	size_t total_num_queires;

        size_t max_depth;
        size_t max_ex_in_leaf;


        bool path_id_feat;

        uint32_t num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;
        bool learn_at_leaf;

        float total_reward;
        bool test_mode;
        float total_test_reward;
        unsigned char Q;
        unsigned char A;

	uint32_t dream_repeats;

        float construct_time;
        float test_time;

	int num_passes;
	bool dream_at_update;

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
            total_reward = 0;
            total_test_reward = 0.;
            construct_time = 0.;
            test_time = 0.;
        }
    };


    float normalized_linear_prod(memory_tree& b, example* ec1, example* ec2)
    {
        if (b.task_id == 1){
            flat_example* fec1 = flatten_sort_example(*b.all, ec1);
            flat_example* fec2 = flatten_sort_example(*b.all, ec2);
            float linear_prod = linear_kernel(fec1, fec2);
            fec1->fs.delete_v(); 
            fec2->fs.delete_v();
            free(fec1);
            free(fec2);
            return linear_prod/pow(fec1->total_sum_feat_sq*fec2->total_sum_feat_sq, 0.5);
        }
        else if (b.task_id != 1){
            double f1_feat_sum_sq = square_norm_feature(ec1->feature_space[b.Q]);
            double f2_feat_sum_sq = square_norm_feature(ec2->feature_space[b.Q]);
            float linear_prod = inner_product_two_features(ec1->feature_space[b.Q], ec2->feature_space[b.Q], f1_feat_sum_sq,f2_feat_sum_sq); //joints.
            return linear_prod;
        }
        return 0;
    }


    void init_tree(memory_tree& b)
    {
        srand48(4000);
        //simple initilization: initilize the root only
        b.routers_used = 0;
        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.nodes[0].base_router = (b.routers_used++);

        b.max_routers = b.max_nodes;
        b.Q = 'J';
        b.A = 'I';

        cout<<"tree initiazliation is done...."<<endl
            <<"max nodes "<<b.max_nodes<<endl
            <<"tree size: "<<b.nodes.size()<<endl
            <<"learn at leaf: "<<b.learn_at_leaf<<endl
            <<"Taks id: "<<b.task_id<<endl
            <<"train_N: "<<b.train_N<<endl
            <<"b.Q: "<<b.Q<<endl
            <<"b.A: "<<b.A<<endl;
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
    float train_node(memory_tree& b, base_learner& base, example& ec_0, const uint32_t cn)
    {
        //predict, learn and predict
        //note: here we first train the router and then predict.
        example& ec = calloc_or_throw<example>();
        if (b.task_id == 1){
            copy_example_data(&ec, &ec_0, false);
        }
        else if (b.task_id != 1){
            copy_example_data(&ec, &ec_0, true); //no feat is true here
            ec.indices.push_back(b.Q);
            ec.feature_space[b.Q].deep_copy_from(ec_0.feature_space[b.Q]); //joints 
            //dst->feature_space[c].deep_copy_from(src->feature_space[c]);
        }

        //MULTICLASS::label_t mc = ec.l.multi;
        //uint32_t save_multi_pred = ec.pred.multiclass;
        ec.l.simple = {FLT_MAX, 1.f, 0.};
        base.predict(ec, b.nodes[cn].base_router);
        float prediction = ec.pred.scalar; 
        float imp_weight = 1.f; //no importance weight.
        
        float weighted_value = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr)/log(2.)+b.alpha*prediction;
        float route_label = weighted_value < 0.f ? -1.f : 1.f;
        
        imp_weight = abs(weighted_value);
        ec.l.simple = {route_label, 1.f, 0.f};
	ec.weight = 1.f;
        base.learn(ec, b.nodes[cn].base_router); //update the router according to the new example.
        
        base.predict(ec, b.nodes[cn].base_router);
        float save_binary_scalar = ec.pred.scalar;
        //ec.l.multi = mc;
        //ec.pred.multiclass = save_multi_pred;

        free_example(&ec);
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
            example& tmp_ec = calloc_or_throw<example>();
            if (b.task_id == 1)
                copy_example_data(&tmp_ec, b.examples[ec_pos]);
            else if (b.task_id != 1){
                copy_example_data(&tmp_ec, b.examples[ec_pos], true);
                tmp_ec.indices.push_back(b.Q);
                tmp_ec.feature_space[b.Q].deep_copy_from(b.examples[ec_pos]->feature_space[b.Q]);
            }
            
            //MULTICLASS::label_t mc = tmp_ec.l.multi;//b.examples[ec_pos]->l.multi;
            //uint32_t save_multi_pred = tmp_ec.pred.multiclass; //b.examples[ec_pos]->pred.multiclass;

            tmp_ec.l.simple = {FLT_MAX, 1.f, 0.f};//b.examples[ec_pos]->l.simple = {1.f, 1.f, 0.f};
            base.predict(tmp_ec, b.nodes[cn].base_router);//base.predict(*b.examples[ec_pos], b.nodes[cn].base_router); //re-predict
            float scalar = tmp_ec.pred.scalar;//b.examples[ec_pos]->pred.scalar;
            if (scalar < 0){
                b.nodes[left_child].examples_index.push_back(ec_pos);
                float leaf_pred = train_node(b, base, *b.examples[ec_pos], left_child);
                descent(b.nodes[left_child], leaf_pred); //fake descent, only for update nl and nr                
            }
            else{
                b.nodes[right_child].examples_index.push_back(ec_pos);
                float leaf_pred = train_node(b, base, *b.examples[ec_pos], right_child);
                descent(b.nodes[right_child], leaf_pred); //fake descent. for update nr and nl
            }
            free_example(&tmp_ec);
            //tmp_ec.l.multi = mc;//b.examples[ec_pos]->l.multi = mc;
            //tmp_ec.pred.multiclass = save_multi_pred;//b.examples[ec_pos]->pred.multiclass = save_multi_pred;
        }
        b.nodes[cn].examples_index.delete_v(); //empty the cn's example list
        b.nodes[cn].nl = std::max(double(b.nodes[left_child].examples_index.size()), 0.001); //avoid to set nl to zero
        b.nodes[cn].nr = std::max(double(b.nodes[right_child].examples_index.size()), 0.001); //avoid to set nr to zero

        if (std::max(b.nodes[cn].nl, b.nodes[cn].nr) > b.max_ex_in_leaf){
            b.max_ex_in_leaf = std::max(b.nodes[cn].nl, b.nodes[cn].nr);
            cout<<b.max_ex_in_leaf<<endl;
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
                    diag_kronecker_product_test(ec, *b.examples[loc], *kprod_ec, b.task_id);
                    //kprod_ec->l.simple = {FLT_MAX, 1., 0.*(-tmp_s+0.5)};
                    kprod_ec->l.simple = {FLT_MAX, 1., tmp_s};
                    base.predict(*kprod_ec, b.max_routers);
                    //score = kprod_ec->pred.scalar;
                    score = kprod_ec->partial_prediction;
                    free_example(kprod_ec);
                }
                else
                    score = normalized_linear_prod(b, &ec, b.examples[loc]); //task 1: innner product between features. task 2: inner product between Qs.
                
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

    float get_reward(memory_tree& b, example& ec, example& retrieved_ec){
        float reward = 0.;
        if (b.task_id == 1)
        {    //multi-class:
            if (ec.l.multi.label == retrieved_ec.l.multi.label)
                reward = 1.;
            else
                reward = 0.;
        }
        else if (b.task_id == 2){ //J and I
            reward = compute_similarity_under_namespace(ec, retrieved_ec, b.A);  //use the image part to compute the reward.
        }
        else if (b.task_id == 3){
            reward = bleu(ec.feature_space[b.A].indicies, retrieved_ec.feature_space[b.A].indicies,2);
        }
        return reward;
    }

    void learn_at_leaf_random(memory_tree& b, base_learner& base, const uint32_t& leaf_id, example& ec)
    {
	    b.total_num_queires++;
	    int32_t ec_id = -1;
	    float reward = 0.f;
	    if (b.nodes[leaf_id].examples_index.size() > 0){
	    	uint32_t pos = uint32_t(merand48(b.all->random_state) * b.nodes[leaf_id].examples_index.size());
		ec_id = b.nodes[leaf_id].examples_index[pos];
	    }
	    if (ec_id != -1){
		float score = normalized_linear_prod(b, &ec, b.examples[ec_id]);
	    	reward = get_reward(b, ec, *b.examples[ec_id]);
		example* kprod_ec = &calloc_or_throw<example>();
		diag_kronecker_product_test(ec, *b.examples[ec_id], *kprod_ec, b.task_id);
		kprod_ec->l.simple = {reward, 1.f, -score};
		kprod_ec->weight = 1.f;
		base.learn(*kprod_ec, b.max_routers);
		free_example(kprod_ec);
	    }
	    return;
    }


    
    void learn_similarity_at_leaf(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        for (uint32_t loc : b.nodes[cn].examples_index)
        {
            example* ec_loc = b.examples[loc];
            float score = normalized_linear_prod(b, &ec, ec_loc); //it split out captions.
            //score = 0.;
            float reward = get_reward(b, ec, *ec_loc);   //reward from 0-1
            example* kprod_ec = &calloc_or_throw<example>();
            diag_kronecker_product_test(ec, *ec_loc, *kprod_ec, b.task_id); //it splits out captions.
     
            float label = ((reward - 0.5f) > 0.f ? 1:-1);
            kprod_ec->l.simple = {label, 1.f, -score};
	    //kprod_ec->l.simple = {reward, 1.f, -score};
	    //kprod_ec->l.simple = {reward-0.5, 1., -score};
            //kprod_ec->l.simple = {label, 1., 0.*(score-0.5)}; //do regression on reward. 
            //kprod_ec->l.simple = {label, 1., -score};
            //if (ec.l.multi.label == b.examples[loc]->l.multi.label) //reward = 1:    
            //    kprod_ec->l.simple = {1., 1., -score};
            //else
                //kprod_ec->l.simple = {-1., 1., score}; //reward = 0:
                //kprod_ec->l.simple = {-0., 1., -score};
        
            base.learn(*kprod_ec, b.max_routers);
            free_example(kprod_ec);
        }
    }
    

    void predict(memory_tree& b, base_learner& base, example& test_ec)
    {
        example& ec = calloc_or_throw<example>();
        if (b.task_id == 1)
            copy_example_data(&ec, &test_ec);
        else if (b.task_id != 1){
            copy_example_data(&ec, &test_ec, true);
            ec.indices.push_back(b.Q);
            ec.feature_space[b.Q].deep_copy_from(test_ec.feature_space[b.Q]); //use joints.
        }
        //remove_repeat_features_in_ec(ec);
        
        //MULTICLASS::label_t mc = ec.l.multi;
        //uint32_t save_multi_pred = ec.pred.multiclass;
        uint32_t cn = 0;
        ec.l.simple = {FLT_MAX, 1.f, 0.};
        while(b.nodes[cn].internal == 1){ 
            //if it's internal
            base.predict(ec, b.nodes[cn].base_router);
            uint32_t newcn = ec.pred.scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right; //do not need to increment nl and nr.
            cn = newcn;
        }
        //ec.l.multi = mc; 
        //ec.pred.multiclass = save_multi_pred;

        int64_t closest_ec = pick_nearest(b, base, cn, ec);
        if (closest_ec != -1){
            float reward = get_reward(b, test_ec, *b.examples[closest_ec]);
            ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            test_ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            test_ec.loss = -reward * test_ec.weight;
            b.total_reward += reward;

	    /*
	    if (b.iter <= b.train_N && b.learn_at_leaf == true){
	    	float score = normalized_linear_prod(b, &ec, b.examples[closest_ec]); //for Q&A task, score is from the inner produce of namespace Q.
		example* kprod_ec = &calloc_or_throw<example>();
		diag_kronecker_product_test(ec, *b.examples[closest_ec], *kprod_ec, b.task_id);
		float label = ((reward - 0.5f) > 0.f ? 1:-1);
		kprod_ec->l.simple = {label, abs(reward-0.5f), -score};
		kprod_ec->weight = abs(reward - 0.5f);
		base.learn(*kprod_ec, b.max_routers);
		free_example(kprod_ec);
	    }*/
		
            if (b.iter > b.train_N)
                b.total_test_reward += reward;
        }
        else{
            b.total_reward += 0.0;
            test_ec.loss = -1.*test_ec.weight;
        }

        free_example(&ec);
    }


    float return_reward_from_node(memory_tree& b, base_learner& base, uint32_t cn, example& test_ec){
       example& ec = calloc_or_throw<example>(); //extract Q from test_ec
       if (b.task_id == 1)
	       copy_example_data(&ec, &test_ec);
       else if (b.task_id != 1){
	       copy_example_data(&ec, &test_ec, true);
	       ec.indices.push_back(b.Q);
	       ec.feature_space[b.Q].deep_copy_from(test_ec.feature_space[b.Q]); //use joints.
       }

    	while(b.nodes[cn].internal != -1){
		base.predict(ec, b.nodes[cn].base_router);
		float prediction = ec.pred.scalar;
		cn = prediction < 0 ? b.nodes[cn].left : b.nodes[cn].right;
	}
	int64_t closest_ec = 0;
	closest_ec = pick_nearest(b,base,cn, ec);  //location at b.examples.

	float reward = 0.f;
	if (closest_ec != -1)
		reward = get_reward(b, test_ec, *b.examples[closest_ec]);//use test_ec here, as querying reward uses features in namespace A.
	b.total_num_queires ++;
	free_example(&ec);
	
	if (b.learn_at_leaf == true && closest_ec != -1)
		learn_similarity_at_leaf(b, base, cn, test_ec);
	/*
	if (b.learn_at_leaf == true && closest_ec != -1){
		float score = normalized_linear_prod(b, &test_ec, b.examples[closest_ec]); //it will split out captions features.
		//score = 0;
		example* kprod_ec = &calloc_or_throw<example>();
		diag_kronecker_product_test(test_ec, *b.examples[closest_ec], *kprod_ec, b.task_id);
		kprod_ec->l.simple = {reward, 1.f, -score};
		kprod_ec->weight = 1.f; //weight*abs(reward - 0.5f);
		base.learn(*kprod_ec, b.max_routers);
		free_example(kprod_ec);
	}*/
	return reward;
    }

    void route_to_leaf(memory_tree& b, base_learner& base, const uint32_t & ec_array_index, uint32_t cn, v_array<uint32_t>& path, bool insertion){
	example& test_ec = *b.examples[ec_array_index]; //test_ec contains Q&A
	example& ec = calloc_or_throw<example>(); //only using Q namespace for routing.
	if(b.task_id == 1)   
		copy_example_data(&ec, &test_ec);
	else if (b.task_id != 1){
		copy_example_data(&ec, &test_ec, true);
		ec.indices.push_back(b.Q);
		ec.feature_space[b.Q].deep_copy_from(test_ec.feature_space[b.Q]); //only use Q
	}

	path.erase();
	while(b.nodes[cn].internal != -1){
		path.push_back(cn);  
		base.predict(ec, b.nodes[cn].base_router);
		float prediction = ec.pred.scalar;
		if (insertion == false)
			cn = prediction < 0 ? b.nodes[cn].left : b.nodes[cn].right;
		else
			cn = descent(b.nodes[cn], prediction);
	}
	path.push_back(cn);

	if (insertion == true){
		b.nodes[cn].examples_index.push_back(ec_array_index);
		if ((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 < b.max_nodes))
			split_leaf(b,base,cn);
	}
    }

    void single_query_and_learn(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, example& test_ec){
    	//ec contains Q and A
    	v_array<uint32_t> path_to_leaf = v_init<uint32_t>();
	route_to_leaf(b, base, ec_array_index, 0, path_to_leaf, false);
	if (path_to_leaf.size() > 1){
		uint32_t random_pos = merand48(b.all->random_state)*(path_to_leaf.size());
		uint32_t cn = path_to_leaf[random_pos];
		if (b.nodes[cn].internal != -1){
			float objective = 0.f;
			float prob_right = 0.5;
			float coin = merand48(b.all->random_state) < prob_right ? 1.f : -1.f;
			if (coin == -1.f){ //go left
				float reward_left_subtree = return_reward_from_node(b,base, b.nodes[cn].left, test_ec);
				objective = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(-reward_left_subtree/(1.-prob_right))/2.;
			}
			else{
				float reward_right_subtree= return_reward_from_node(b,base, b.nodes[cn].right, test_ec);
				objective = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr) + b.alpha*(reward_right_subtree/prob_right)/2.;
			}
			example& ec = calloc_or_throw<example>(); //extract Q (caption feature) from test_ec
			if (b.task_id == 1)
				copy_example_data(&ec, &test_ec);
			else if (b.task_id != 1){
				copy_example_data(&ec, &test_ec, true);
				ec.indices.push_back(b.Q); //ec contains captions of images
				ec.feature_space[b.Q].deep_copy_from(test_ec.feature_space[b.Q]); //use joints.
			}
			ec.weight = fabs(objective);
			if (ec.weight >= 100.f)
				ec.weight = 100.f;
			else if (ec.weight < 0.01)
				ec.weight = 0.01f;
			//ec.weight=1.f;
			ec.l.simple = {objective < 0. ? -1.f : 1.f, 1.f, 0.};
			base.learn(ec, b.nodes[cn].base_router); //ec only contains caption features.
			free_example(&ec);
		}
		else{
			if(b.learn_at_leaf == true)
				//learn_at_leaf_random(b, base, cn, test_ec);
				learn_similarity_at_leaf(b, base, cn, test_ec);
		}
	}
	path_to_leaf.delete_v();
    }

    void insert_example_hal(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, example& ec) {
	//here ec contains both Q and A.
    	//multiple_query_learn_and_final_insert(b, base, ec_array_index, ec);
	single_query_and_learn(b, base, ec_array_index, ec);
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
	     //learn_at_leaf_random(b, base, cn, *b.examples[ec_array_index]);
             learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]);

        if((b.nodes[cn].internal == -1) && (fake_insert == false)) //get to leaf:
        {   
            b.nodes[cn].examples_index.push_back(ec_array_index);
            if (b.nodes[cn].examples_index.size() > b.max_ex_in_leaf)
            {
                b.max_ex_in_leaf = b.nodes[cn].examples_index.size();
            }
            float leaf_pred = train_node(b, base, *b.examples[ec_array_index], cn); //tain the leaf as well.
            descent(b.nodes[cn], leaf_pred); //this is a faked descent, the purpose is only to update nl and nr of cn

            //if the number of examples exceeds the max_leaf_examples, and not reach the max_nodes - 2 yet, we split:
            if((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 <= b.max_nodes)){
                split_leaf(b, base, cn); 
            }
        }
    }

    void experience_replay(memory_tree& b, base_learner& base)
    {
        uint32_t cn = 0; //start from root, randomly descent down! 
        uint32_t ec_id = random_sample_example_pop(b,cn);
        if (ec_id >= 0){
	    if (b.iter < b.train_N)
            	insert_example(b, base, ec_id); 
	    else if (b.iter == b.train_N){
		if (b.dream_at_update == false){
	    		v_array<uint32_t> tmp_path = v_init<uint32_t>();
			route_to_leaf(b, base, ec_id, 0, tmp_path, true);
			tmp_path.delete_v();
		}
		else
			insert_example(b, base, ec_id);
	    }
        }

    }

    //learn: descent the example from the root while generating binary training
    //example for each node, including the leaf, and store the example at the leaf.
    void learn(memory_tree& b, base_learner& base, example& ec)
    {   
        //int32_t train_N = 900*2;
        //int32_t train_N = 173551*2;
        //int32_t train_N = 7000*2;
        b.iter++;       
        //if (b.test_mode == false){
        if (b.iter<=b.train_N){
            predict(b, base, ec);
            if (b.iter%5000 == 0)
                //cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
                cout<<"at iter "<<b.iter<<", average reward: "<<b.total_reward*1./b.iter<<endl;

            clock_t begin = clock();
            example* new_ec = &calloc_or_throw<example>();
            copy_example_data(new_ec, &ec);
            //remove_repeat_features_in_ec(*new_ec); ////sort unique.
            b.examples.push_back(new_ec);   //memory new_ec contains both Q and A
            b.num_ecs++; 

            insert_example(b, base, b.examples.size()-1);
            for (uint32_t i = 0; i < b.dream_repeats; i++)
                experience_replay(b, base);   
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            b.construct_time += elapsed_secs;
        }
	if (b.iter == b.train_N){
		cout<<"################################"<<endl;
		//multiple passes with reward based learning0
		for (int pass = 0; pass < b.num_passes-1; pass++){
			cout<< "#### AT Pass: "<<pass+1<<", with number of memories: "<<b.examples.size()<<endl;
			for (size_t ec_id = 0; ec_id < b.examples.size(); ec_id++){
				predict(b, base, *b.examples[ec_id]);
				if ((b.iter + ec_id) % 1000 == 0)
					cout<<"at iter "<<b.iter+pass*b.examples.size() + ec_id<<", average reward: "<<b.total_reward*1./(b.iter+ec_id+pass*b.examples.size())<<", total number of reward queries: "<<b.total_num_queires<<endl;
				insert_example_hal(b, base, ec_id, *b.examples[ec_id]);

				for(uint32_t i = 0; i < b.dream_repeats; i++)
					experience_replay(b, base);
			}
		}
	}

        //else if (b.test_mode == true){
        if (b.iter > b.train_N){
            if ((b.iter-b.train_N) % 1000 == 0)
                cout<<"at iter "<<b.iter-b.train_N<<",total_reward "<<b.total_test_reward<<", avg_reward: "<<b.total_test_reward*1./(b.iter-b.train_N)<<endl;
            clock_t begin = clock();
            predict(b, base, ec);
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            b.test_time += elapsed_secs;
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
        cout<<"average test reward: "<<b.total_test_reward/(b.iter - b.train_N)<<", in total of "<<b.iter-b.train_N<<" test examples."<<endl;
        cout<<"construct time: "<<b.construct_time<<endl;
        cout<<"test time: "<<b.test_time<<endl;
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
            writeit(b.Q, "Q");
            writeit(b.A, "A");
            writeit(b.task_id, "Task id");
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
      ("learn_at_leaf", po::value<bool>()->default_value(true), "whether or not learn at leaf (defualt = True)")
      ("task", po::value<uint32_t>()->default_value(1.), "task: 1:multiclass; 2: Robot Configuration; 3:Q&A with bleu as reward")
      ("train_N", po::value<size_t>()->default_value(1000), "num of training examples")
      ("dream_repeats", po::value<uint32_t>()->default_value(1.), "number of dream operations per example (default = 1)")
      ("dream_at_update", po::value<bool>()->default_value(false), "turn on dream operations at reward based update as well")
      ("num_passes", po::value<int>()-> default_value(1), "number of passes (note this is different from vw's passes parameter)")
      ("Alpha", po::value<float>()->default_value(0.1), "Alpha");
     add_options(all);

    po::variables_map& vm = all.vm;
    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.all = &all;
    tree.max_nodes = vm["memory_tree"].as<uint32_t>();
    tree.learn_at_leaf = vm["learn_at_leaf"].as<bool>();
    tree.bandit = true;

    if (vm.count("leaf_example_multiplier"))
      {
	tree.max_leaf_examples = vm["leaf_example_multiplier"].as<uint32_t>() * (log(tree.max_nodes)/log(2));
	*all.file_options << " --leaf_example_multiplier " << vm["leaf_example_multiplier"].as<uint32_t>();
      }
    if (vm.count("num_passes")){
    	tree.num_passes = vm["num_passes"].as<int>();
	*all.file_options << " --num_passes "<<vm["num_passes"].as<int>();
    }

    if (vm.count("dream_repeats")){
    	tree.dream_repeats = vm["dream_repeats"].as<uint32_t>();
	*all.file_options << " --dream_repeats "<<vm["dream_repeats"].as<uint32_t>();
    }

    if (vm.count("task"))
    {
        tree.task_id = vm["task"].as<uint32_t>();
        *all.file_options << " --task "<< vm["task"].as<uint32_t>();
    }
    if (vm.count("dream_at_update")){
    	tree.dream_at_update = vm["dream_at_update"].as<bool>();
	*all.file_options << " --dream_at_update "<<vm["dream_at_update"].as<bool>();
    }
    if (vm.count("train_N"))
    {
        tree.train_N = vm["train_N"].as<size_t>();
        *all.file_options << " --train_N "<< vm["train_N"].as<size_t>();
    }

    if (vm.count("Alpha"))
      {
	tree.alpha = vm["Alpha"].as<float>();
	*all.file_options << " --Alpha " << tree.alpha;
      }
    
    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "memory_tree:" << " "
                    <<"max_nodes = "<< tree.max_nodes << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
                    <<"alpha = "<<tree.alpha<<" "
                    <<"Task = "<<tree.task_id
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
*/
